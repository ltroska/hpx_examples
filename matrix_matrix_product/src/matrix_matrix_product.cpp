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
  
std::vector<double> multiply(hpx::future<block_data> A_fut,hpx::future<block_data> x_fut,
    boost::uint64_t block_rows,
    boost::uint64_t block_columns, boost::uint64_t tile_size)
{    
    const block_data A(A_fut.get());
    const block_data x(x_fut.get());
    
    std::vector<double> rhs(block_rows, 0);
            
    if (tile_size < block_columns)
    {
        //TODO: maybe += tilesize here too
        for (boost::uint64_t i = 0; i < block_rows; ++i)
        {
            for (boost::uint64_t j = 0; j < block_columns; j += tile_size)
            {
                boost::uint64_t max_j = std::min(block_columns, j + tile_size);

                for (boost::uint64_t jt = j; jt != max_j; ++jt)
                {
                   rhs[i] += A[jt * block_rows + i] * x[j];
                }
            }
        }
    }        
    else
    {
        for (boost::uint64_t i = 0; i < block_rows; ++i)
        {
            for (boost::uint64_t j = 0; j < block_columns; ++j)
            {      
                rhs[i] += A[j * block_rows + i] * x[j];
            }
            
        }
    }
    
    return rhs;
}
  

int hpx_main(boost::program_options::variables_map& vm)
{
    hpx::id_type here = hpx::find_here();
    bool root = (here == hpx::find_root_locality());

    boost::uint64_t num_localities = hpx::get_num_localities().get();

    boost::uint64_t num_rows = vm["matrix_size"].as<boost::uint64_t>();
    boost::uint64_t num_columns = vm["matrix_size"].as<boost::uint64_t>();
    boost::uint64_t iterations = vm["iterations"].as<boost::uint64_t>();
    boost::uint64_t num_blocks = vm["num_blocks"].as<boost::uint64_t>();
    boost::uint64_t tile_size = num_columns;
    
    boost::uint64_t block_rows = num_rows / num_localities / num_blocks;
    boost::uint64_t block_columns = num_columns / num_localities / num_blocks;
    boost::uint64_t block_size = block_rows * num_columns;    
    
    boost::uint64_t num_total_blocks = num_blocks * num_localities;
    
    double low = vm["low"].as<double>();
    double high = vm["high"].as<double>();
    
    boost::uint64_t bytes =
        static_cast<boost::uint64_t>(sizeof(double) * (num_rows * num_columns + 2 * num_rows));
    
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
