//  Copyright (c) 2016 Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <boost/range/irange.hpp>
#include <hpx/include/parallel_algorithm.hpp>

#include <random>

bool verbose = false;

char const* A_block_basename = "/matrix_vector_product/block/A";
char const* B_block_basename = "/matrix_vector_product/block/B";
char const* C_block_basename = "/matrix_vector_product/block/C";

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
        return block_data(&data_[offset], size);;
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
    
    hpx::future<block_data> get_data(boost::uint64_t offset, boost::uint64_t size)
    {
        block_component::get_data_action act;
        return hpx::async(act, get_id(), offset, size);
    }
};
      
typedef hpx::components::component<block_component> block_component_type;
HPX_REGISTER_COMPONENT(block_component_type, block_component);

typedef block_component::get_data_action get_data_action;
HPX_REGISTER_ACTION(get_data_action);
  
void multiply(hpx::future<block_data> A_fut, hpx::future<block_data> B_fut,
    hpx::future<block_data> C_fut, boost::uint64_t block_rows,
    boost::uint64_t num_columns, boost::uint64_t phase)
{    
    const block_data A(A_fut.get());
    const block_data B(B_fut.get());
    block_data C(C_fut.get());

    for (boost::uint64_t k = 0; k < block_rows; ++k)
    {
        for (boost::uint64_t i = 0; i < block_rows; ++i)
        {        
            double temp = 0;
    
            for (boost::uint64_t j = 0; j < num_columns; ++j)
            {      
                temp +=
                    A[k * num_columns + j] * B[i * num_columns + j];                    
            }
            
            C[i * block_rows + k] = temp;
        }
    }        
}
  

int hpx_main(boost::program_options::variables_map& vm)
{
    hpx::id_type here = hpx::find_here();
    bool root = (here == hpx::find_root_locality());

    const boost::uint64_t num_localities = hpx::get_num_localities().get();

    const boost::uint64_t num_rows = vm["matrix_size"].as<boost::uint64_t>();
    const boost::uint64_t num_columns = vm["matrix_size"].as<boost::uint64_t>();
    const boost::uint64_t iterations = vm["iterations"].as<boost::uint64_t>();
    const boost::uint64_t num_blocks = vm["num_blocks"].as<boost::uint64_t>();
    boost::uint64_t tile_size = num_columns;
    
    const boost::uint64_t block_rows = num_rows / num_localities / num_blocks;
    const boost::uint64_t block_size = block_rows * num_columns;
    
    const boost::uint64_t order = block_rows * block_rows;
    
    const boost::uint64_t num_total_blocks = num_blocks * num_localities;
    
    const double low = vm["low"].as<double>();
    const double high = vm["high"].as<double>();
    
    boost::uint64_t bytes =
        static_cast<boost::uint64_t>(sizeof(double) * (3 * num_rows * num_columns));
    
    if(vm.count("tile_size"))
        tile_size = vm["tile_size"].as<boost::uint64_t>();

    verbose = vm.count("verbose") ? true : false;

    if(root)
    {
        std::cout
            << "Matrix matrix product: A*B = C with random A and B\n"
            << "Matrix dimensions:\t\t" << num_rows << "x" << num_columns << "\n"
            << "Number of blocks:\t\t" << num_blocks << "\n"
            << "Local block dimension:\t\t"     << block_rows << "x"
                                                << num_columns << "\n"
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
    std::vector<block> B(num_total_blocks);
    std::vector<block> C(num_total_blocks);
    
    boost::uint64_t local_blocks_begin = id * num_blocks;
    boost::uint64_t local_blocks_end = (id + 1) * num_blocks;
    
    for (boost::uint64_t b = 0; b != num_blocks; ++b)
    {
        boost::uint64_t block_index = b + local_blocks_begin;
        
        A[block_index] = block(block_index, block_size, A_block_basename);
        B[block_index] = block(block_index, block_size, B_block_basename);
        C[block_index] = block(block_index, block_size, C_block_basename);
    }
    
    std::vector<hpx::future<hpx::id_type> > A_ids =
        hpx::find_all_from_basename(A_block_basename, num_total_blocks);
        
    std::vector<hpx::future<hpx::id_type> > B_ids =
        hpx::find_all_from_basename(B_block_basename, num_total_blocks);
        
    std::vector<hpx::future<hpx::id_type> > C_ids =
        hpx::find_all_from_basename(C_block_basename, num_total_blocks);
        
    auto range = boost::irange(local_blocks_begin, local_blocks_end);
    
    hpx::parallel::for_each(
        hpx::parallel::par, boost::begin(range), boost::end(range),
        [&](boost::uint64_t b)
        {             
            rand_double rd(low, high);
        
            std::shared_ptr<block_component> A_ptr =
                hpx::get_ptr<block_component>(A[b].get_id()).get();
                
            std::shared_ptr<block_component> B_ptr =
                hpx::get_ptr<block_component>(B[b].get_id()).get();
                
            std::shared_ptr<block_component> C_ptr =
                hpx::get_ptr<block_component>(C[b].get_id()).get();
                

            for (boost::uint64_t i = 0; i != block_rows; ++i)
            {
                for (boost::uint64_t j = 0; j != num_columns; ++j)
                {
                    A_ptr->data_[i * num_columns + j] = b * block_rows + i;
                    C_ptr->data_[i * num_columns + j] = 0;
                }                        
            }

            for (boost::uint64_t j = 0; j != block_rows; ++j)
            {
                for (boost::uint64_t i = 0; i != num_rows; ++i)
                {
                    B_ptr->data_[j * num_rows + i] = i;
                }                        
            }          

            hpx::register_with_basename(A_block_basename, A[b].get_id(), b);
            hpx::register_with_basename(B_block_basename, B[b].get_id(), b);
            hpx::register_with_basename(C_block_basename, C[b].get_id(), b);
        }
    );
    
    hpx::wait_all(A_ids);
    hpx::wait_all(B_ids);
    hpx::wait_all(C_ids);
    
    for (boost::uint64_t b = 0; b != num_total_blocks; ++b)
    {
        if (b < local_blocks_begin || b >= local_blocks_end)
        {
            A[b] = block(std::move(A_ids[b]));
            B[b] = block(std::move(B_ids[b]));
            C[b] = block(std::move(C_ids[b]));
        }
    }
    
    double avgtime = 0.0;
    double maxtime = 0.0;
    double mintime = 366.0 * 24.0 * 3600.0;
    
    for (boost::uint64_t iter = 0; iter != iterations; ++iter)
    {           
        hpx::util::high_resolution_timer t;
                
        std::vector<hpx::future<void> > block_futures;
            block_futures.resize(num_blocks);
        
        hpx::parallel::for_each(
            hpx::parallel::par, boost::begin(range), boost::end(range),
            [&](boost::uint64_t b)
            {
                 auto phase_range =
                        boost::irange(static_cast<boost::uint64_t>(0), num_total_blocks);
                   
                std::vector<hpx::future<void> > phase_futures;

                   
                for (boost::uint64_t phase : phase_range)
                {       
                   phase_futures.push_back(hpx::dataflow(    
                        &multiply,
                        A[b].get_data(0, block_rows * num_columns),
                        B[phase].get_data(0, num_rows * block_rows),
                        C[b].get_data(phase * order, order),
                        block_rows,
                        num_columns,
                        phase
                   ));       
                }

                block_futures[b - local_blocks_begin] =
                    hpx::when_all(phase_futures);
            }                      
                
        );
                
        hpx::wait_all(block_futures);
                
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
            std::cout << "next block " << b << "\n";        
                    std::shared_ptr<block_component> A_ptr =
                        hpx::get_ptr<block_component>(A[b].get_id()).get();
                    
            
            for (boost::uint64_t i = 0; i != block_rows; ++i)
            {
                for (boost::uint64_t j = 0; j != num_columns; ++j)
                {
                    std::cout << A_ptr->data_[i * num_columns + j] << " ";
                }
                
                std::cout << "\n";
            }
        }            
        std::cout << std::endl;        
        
        for (boost::uint64_t b = local_blocks_begin; b != local_blocks_end; ++b)
        {        
            std::cout << "next block " << b << "\n";        
                    std::shared_ptr<block_component> B_ptr =
                        hpx::get_ptr<block_component>(B[b].get_id()).get();
                    
            
            for (boost::uint64_t i = 0; i != num_rows; ++i)
            {
                for (boost::uint64_t j = 0; j != block_rows; ++j)
                {
                    std::cout << B_ptr->data_[j * num_rows + i] << " ";
                }
                
                std::cout << "\n";
            }
        }            
        std::cout << std::endl;        
        
        for (boost::uint64_t b = local_blocks_begin; b != local_blocks_end; ++b)
        {        
            std::cout << "next block " << b << "\n";        
                    std::shared_ptr<block_component> C_ptr =
                        hpx::get_ptr<block_component>(C[b].get_id()).get();
                    
            
            for (boost::uint64_t i = 0; i != block_rows; ++i)
            {
                for (boost::uint64_t j = 0; j != num_columns; ++j)
                {
                    std::cout << C_ptr->data_[j * block_rows + i] << " ";
                }
                
                std::cout << "\n";
            }
        }            
        std::cout << std::endl;
    }
    
    return hpx::finalize();
}

int main(int argc, char* argv[])
{    
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("matrix_size", value<boost::uint64_t>()->default_value(1024),
         "Number of matrix rows/column")
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
