//  Copyright (c) 2016 Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <boost/range/irange.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/lcos/barrier.hpp>

#include <random>

bool verbose = false;

char const* A_block_basename = "/matrix_vector_product/block/A";
char const* x_block_basename = "/matrix_vector_product/block/x";
char const* rhs_block_basename = "/matrix_vector_product/block/rhs";
char const* barrier_basename = "/matrix_vector_product/barrier";

class rand_double
{
public:
    rand_double(double low, double high)
        :   r(std::bind(std::uniform_real_distribution<>(low,high),
                std::default_random_engine()))
    {}

    double operator()()
    {
        return r();
    }

private:
    std::function<double()> r;
};

struct block_data
{
    enum mode
    {
        reference,
        owning
    };
    
    block_data()
    :   size_(0),
        data_(nullptr),
        mode_(reference)
    {}

    block_data(double* data, boost::uint64_t size)
    :   size_(size),
        data_(data),
        mode_(reference)
    
    {}
    
    ~block_data()
    {
        if (data_ != nullptr && mode_ == owning)
            delete[] data_;
    }
    
    block_data(block_data&& other)
    :   size_(other.size_),
        data_(other.data_),
        mode_(other.mode_)
    {
        if (mode_ == owning)
        {
            other.data_ = nullptr;
            other.size_ = 0;
        }
    }
    
    block_data& operator=(block_data&& other)
    {
        size_ = other.size_;
        data_ = other.data_;
        mode_ = other.mode_;
        
        if (mode_ == owning)
        {
            other.data_ = nullptr;
            other.size_ = 0;
        }
    
        return *this;
    }        
    
    double operator[](std::size_t index) const
    {
        return data_[index];
    }
    
    double& operator[](std::size_t index)
    {
        return data_[index];
    }
    
    void load(hpx::serialization::input_archive & ar, unsigned version)
    {
        ar & size_;
        if(size_ > 0)
        {
            data_ = new double[size_];
            hpx::serialization::array<double> arr(data_, size_);
            ar >> arr;
            mode_ = owning;
        }
    }

    void save(hpx::serialization::output_archive & ar, unsigned version) const
    {
        ar & size_;
        if(size_ > 0)
        {
            hpx::serialization::array<double> arr(data_, size_);
            ar << arr;
        }
    }
    
    HPX_SERIALIZATION_SPLIT_MEMBER()

    boost::uint64_t size_;
    double* data_;
    mode mode_;
    
    HPX_MOVABLE_ONLY(block_data);
};

struct block_component
    :   hpx::components::component_base<block_component>
{
    block_component() {}
    
    block_component(boost::uint64_t size)
        :   data_(size)
    {}
    
    block_data get_data(boost::uint64_t offset, boost::uint64_t size)
    {
        return block_data(&data_[offset], size);
    }
    
    HPX_DEFINE_COMPONENT_DIRECT_ACTION(block_component, get_data);    
    
    std::vector<double> data_;
};

struct block
    :   hpx::components::client_base<block, block_component>
{
    typedef hpx::components::client_base<block, block_component> base_type;
    
    block() {}
    
    block(hpx::future<hpx::id_type>&& id)
        :   base_type(std::move(id))
    {}

    block(boost::uint64_t id, boost::uint64_t size, const char* base_name)
        :   base_type(hpx::new_<block_component>(hpx::find_here(), size))
    {}
    
    hpx::future<block_data> get_data(boost::uint64_t offset,
                                        boost::uint64_t size)
    {
        block_component::get_data_action act;
        return hpx::async(act, get_id(), offset, size);
    }
};
      
typedef hpx::components::component<block_component> block_component_type;
HPX_REGISTER_COMPONENT(block_component_type, block_component);

typedef block_component::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action);
  
std::vector<double> multiply(
    hpx::future<block_data> A_fut,hpx::future<block_data> x_fut,
    boost::uint64_t block_rows,
    boost::uint64_t block_columns, boost::uint64_t tile_size)
{    
   //hpx::util::high_resolution_timer t;
    
    const block_data A(A_fut.get());
    const block_data x(x_fut.get());
    
    std::vector<double> rhs(block_rows, 0);
            
    for (boost::uint64_t i = 0; i < block_rows; ++i)
    {
        for (boost::uint64_t j = 0; j < block_columns; ++j)
        {  
            rhs[i] += A[i * block_columns + j] * x[j];
        }            
    }
    
    //std::cout << t.elapsed() << std::endl;
    return rhs;
}
  

int hpx_main(boost::program_options::variables_map& vm)
{
    const hpx::id_type here = hpx::find_here();
    const bool root = (here == hpx::find_root_locality());

    const boost::uint64_t num_localities = hpx::get_num_localities().get();

	const boost::uint64_t num_rows = vm["rows"].as<boost::uint64_t>();
	const boost::uint64_t num_columns = vm["columns"].as<boost::uint64_t>();
    const boost::uint64_t iterations = vm["iterations"].as<boost::uint64_t>();
    const boost::uint64_t num_blocks = vm["num_blocks"].as<boost::uint64_t>();
    boost::uint64_t tile_size = num_columns;
    
    const boost::uint64_t block_rows = num_rows / num_localities / num_blocks;
    const boost::uint64_t block_columns = num_columns;
    const boost::uint64_t block_size = block_rows * block_columns;     
    
    const boost::uint64_t num_total_blocks = num_blocks * num_localities;
    
    const double low = vm["low"].as<double>();
    const double high = vm["high"].as<double>();
    
    const boost::uint64_t bytes =
        static_cast<boost::uint64_t>(sizeof(double) 
                                    * (num_rows * num_columns + 2 * num_rows));
    
    if(vm.count("tile_size"))
        tile_size = vm["tile_size"].as<boost::uint64_t>();

    verbose = vm.count("verbose") ? true : false;

    if(root)
    {
        std::cout
            << "Matrix vector product: A*x = b with random A and x\n"
            << "Matrix dimensions:\t\t" << num_rows << "x" << num_columns << "\n"
            << "Number of blocks:\t\t" << num_blocks << "\n"
            << "Local block dimension:\t\t"     << block_rows << "x"
                                                << block_columns << "\n"
            << "Number of total blocks:\t\t" << num_total_blocks << "\n"                             
            << "Number of localities:\t\t" << num_localities << "\n";
        if(tile_size < num_columns)
            std::cout << "Tile size:\t\t\t" << tile_size << "\n";
        else
            std::cout << "Tile size:\t\t\tUntiled\n";
        std::cout
            << "Number of iterations\t\t" << iterations << "\n";
    }
    
    boost::uint64_t id = hpx::get_locality_id();
    
    std::vector<block> A(num_total_blocks);
    block x;
    std::vector<block> rhs(num_total_blocks);
    
    boost::uint64_t local_blocks_begin = id * num_blocks;
    boost::uint64_t local_blocks_end = (id + 1) * num_blocks;
    
    for (boost::uint64_t b = 0; b != num_blocks; ++b)
    {
        boost::uint64_t block_index = b + local_blocks_begin;
        
        A[block_index] = block(block_index, block_size, A_block_basename);
        rhs[block_index] = block(block_index, block_rows, rhs_block_basename);
    }
    
    if (root)
        x = block(0, num_columns, x_block_basename);
    
    std::vector<hpx::future<hpx::id_type> > A_ids =
        hpx::find_all_from_basename(A_block_basename, num_total_blocks);
        
    hpx::future<hpx::id_type> x_id =
        hpx::find_from_basename(x_block_basename, 0);
        
    std::vector<hpx::future<hpx::id_type> > rhs_ids =
        hpx::find_all_from_basename(rhs_block_basename, num_total_blocks);
        
    auto range = boost::irange(local_blocks_begin, local_blocks_end);
        
    hpx::parallel::for_each(
        hpx::parallel::par, boost::begin(range), boost::end(range),
        [&](boost::uint64_t b)
        {                
            hpx::register_with_basename(A_block_basename, A[b].get_id(), b);
            hpx::register_with_basename(rhs_block_basename, rhs[b].get_id(), b);
        }
    );
    
    if (root)
        hpx::register_with_basename(x_block_basename, x.get_id(), 0);
    
    hpx::wait_all(A_ids);
    x_id.wait();
    hpx::wait_all(rhs_ids);
    
    for (boost::uint64_t b = 0; b != num_total_blocks; ++b)
    {
        if (b < local_blocks_begin || b >= local_blocks_end)
        {
            A[b] = block(std::move(A_ids[b]));
            rhs[b] = block(std::move(rhs_ids[b]));
        }
    }
    
    x = block(std::move(x_id));
    
    double avgtime = 0.0;
    double maxtime = 0.0;
    double mintime = 366.0 * 24.0 * 3600.0;
    
    rand_double rd(low, high);

    hpx::parallel::for_each(
        hpx::parallel::par, boost::begin(range), boost::end(range),
        [&](boost::uint64_t b)
        {                
                std::shared_ptr<block_component> A_ptr =
                    hpx::get_ptr<block_component>(A[b].get_id()).get();
                                        
                std::shared_ptr<block_component> rhs_ptr =
                    hpx::get_ptr<block_component>(rhs[b].get_id()).get();
                    
                for (boost::uint64_t j = 0; j != num_columns; ++j)
                {
                    for (boost::uint64_t i = 0; i != block_rows; ++i)
                    {
                        A_ptr->data_[i * block_columns + j] = rd();
                    }
                    
                }
                
                for (boost::uint64_t i = 0; i != block_rows; ++i)
                {
                    rhs_ptr->data_[i] = 0;
                }
        }
    );
    
    hpx::lcos::barrier b;
    
    if (root)
    {
        std::shared_ptr<block_component> x_ptr =
                    hpx::get_ptr<block_component>(x.get_id()).get();
                    
        for (boost::uint64_t i = 0; i != num_columns; ++i)
            x_ptr->data_[i] = rd();  
    
        b = std::move(hpx::lcos::barrier::create(hpx::find_here(), num_localities));
        hpx::agas::register_name_sync(barrier_basename, b.get_id());
    }
    else
    {
        hpx::id_type idb = hpx::agas::on_symbol_namespace_event(
                barrier_basename, hpx::agas::symbol_ns_bind, true).get();
        b = std::move(hpx::lcos::barrier(idb));
    }
        
    b.wait();
    
    for (boost::uint64_t iter = 0; iter != iterations; ++iter)
    {   
         hpx::parallel::for_each(
            hpx::parallel::par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b)
            {                
                    std::shared_ptr<block_component> rhs_ptr =
                        hpx::get_ptr<block_component>(rhs[b].get_id()).get();

                    
                    for (boost::uint64_t i = 0; i != block_rows; ++i)
                    {
                        rhs_ptr->data_[i] = 0;
                    }
            }
        );
        
        hpx::util::high_resolution_timer t;
               
        hpx::parallel::for_each(
            hpx::parallel::par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b)
            {
                std::shared_ptr<block_component> rhs_ptr =
                        hpx::get_ptr<block_component>(rhs[b].get_id()).get();
                
            
               hpx::dataflow(    
                    &multiply,
                    A[b].get_data(0, block_size),
                    x.get_data(0, num_columns),
                    block_rows,
                    block_columns,
                    tile_size
                ).then(
                    hpx::util::unwrapped(
                        [&](std::vector<double> r)
                        {
                            std::copy(r.begin(), r.end(), rhs_ptr->data_.begin());
                        }
                    )
                ).wait();                                                
            }            
        );
                
        b.wait();
        
        double elapsed = t.elapsed();
        
        if (verbose)
            std::cout
                << "Iteration " << iter << " took " << elapsed << "(s)"
                << std::endl;
        
        if (iter > 0 || iterations == 1)
        {
            avgtime = avgtime + elapsed;
            maxtime = std::max(maxtime, elapsed);
            mintime = std::min(mintime, elapsed);
        }
    }
    
    if (root)
    {
        avgtime = avgtime/static_cast<double>(
                    (std::max)(iterations-1, static_cast<boost::uint64_t>(1)));
                    
        std::cout
            << "Rate (MB/s):\t" << 1.e-6 * bytes/mintime << "\n"
            << "Avg time (s):\t" << avgtime << "\n"
            << "Min time (s):\t" << mintime << "\n"
            << "Max time (s):\t" << maxtime << "\n"
            << std::endl;
    } 
    
    if (verbose)
    {
        for (boost::uint64_t b = local_blocks_begin; b != local_blocks_end; ++b)
        {        
            std::cout << "Checking block " << b << ": ";        
                    std::shared_ptr<block_component> A_ptr =
                        hpx::get_ptr<block_component>(A[b].get_id()).get();
                                                
            std::shared_ptr<block_component> rhs_ptr =
                hpx::get_ptr<block_component>(rhs[b].get_id()).get();
            
	
			bool correct = true; 

	 		for (boost::uint64_t i = 0; i != block_rows; ++i)
            {
				double res = 0;
				
				block_data x_data = x.get_data(0, num_columns).get();
                
                for (boost::uint64_t j = 0; j != num_columns; ++j)
                    res += A_ptr->data_[i * block_columns + j] * x_data[j];

				if (std::abs(res - rhs_ptr->data_[i]) > 1e-9)
{
                    std::cout << res << " " << rhs_ptr->data_[i] << std::endl;
					correct = false;
}
			}

			std::cout << "The locally computed data is ";
			if (correct)
				std::cout << "correct.";
			else
				std::cout << "wrong.";
			std::cout << std::endl;

        }
                    
        std::cout << std::endl;
    }
    
    b.wait();
    
    return hpx::finalize();
}

int main(int argc, char* argv[])
{    
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("matrix_size", value<boost::uint64_t>()->default_value(1024),
         "Number of matrix rows/column")
        ("rows", value<boost::uint64_t>()->default_value(1024),
		 "Number of matrix rows")
		("columns", value<boost::uint64_t>()->default_value(1024),
		 "Number of matrix columns")
        ("iterations", value<boost::uint64_t>()->default_value(10),
         "Number of iterations to run for")
        ("num_blocks", value<boost::uint64_t>()->default_value(1),
         "Number of blocks to divide the rows into")
        ("tile_size", value<boost::uint64_t>(),
         "Number of tiles to divide the blocks into")
        ("low", value<double>()->default_value(-100),
         "Lower bound for the random nummer generator to fill the matrix/vector")
        ("high", value<double>()->default_value(100),
         "Upper bound for the random nummer generator to fill the matrix/vector")      
        ( "verbose", "Verbose output")
    ;

    // hpx_main needs to run on all localities this main is called on
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    return hpx::init(desc_commandline, argc, argv, cfg);
}
