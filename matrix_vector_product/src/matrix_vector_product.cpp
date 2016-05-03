//  Copyright (c) 2016 Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

bool verbose = true;

int hpx_main(boost::program_options::variables_map& vm)
{
    hpx::id_type here = hpx::find_here();
    bool root = (here == hpx::find_root_locality());

    boost::uint64_t num_localities = hpx::get_num_localities().get();

    boost::uint64_t num_rows = vm["matrix_rows"].as<boost::uint64_t>();
    boost::uint64_t num_columns = vm["matrix_columns"].as<boost::uint64_t>();
    boost::uint64_t iterations = vm["iterations"].as<boost::uint64_t>();
    boost::uint64_t num_local_blocks = vm["num_blocks"].as<boost::uint64_t>();
    boost::uint64_t tile_size = num_columns;
    
    boost::uint64_t block_rows = num_rows / num_localities / num_local_blocks;
    boost::uint64_t block_columns = num_columns / num_local_blocks;
    
    boost::uint64_t num_blocks = num_local_blocks * num_localities; 
    
    if(vm.count("tile_size"))
        tile_size = vm["tile_size"].as<boost::uint64_t>();

    //verbose = vm.count("verbose") ? true : false;
    
    if(root)
    {
        std::cout
            << "Matrix vector product: A*x = b with random A and x\n"
            << "Matrix dimensions:\t\t" << num_rows << "x" << num_columns << "\n"
            << "Number of local blocks:\t\t" << num_local_blocks << "\n"
            << "Local block dimension:\t\t"     << block_rows << "x"
                                                << block_columns << "\n"
            << "Number of total blocks:\t\t" << num_blocks << "\n"                             
            << "Number of localities:\t\t" << num_localities << "\n";
        if(tile_size < num_columns)
            std::cout << "Tile size:\t\t\t" << tile_size << "\n";
        else
            std::cout << "Tile size:\t\t\tUntiled\n";
        std::cout
            << "Number of iterations\t\t" << iterations << "\n";
    }
    
    return hpx::finalize();
}

int main(int argc, char* argv[])
{    
    using namespace boost::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()
        ("matrix_rows", value<boost::uint64_t>()->default_value(10),
         "Number of matrix rows")
         ("matrix_columns", value<boost::uint64_t>()->default_value(10),
         "Number of matrix columns (consequently also vector entries)")
        ("iterations", value<boost::uint64_t>()->default_value(10),
         "Number of iterations to run for")
        ("num_blocks", value<boost::uint64_t>()->default_value(1),
         "Number of blocks to divide the rows into")
        ("tile_size", value<boost::uint64_t>(),
         "Number of tiles to divide the blocks into")       
        ( "verbose", "Verbose output")
    ;

    // hpx_main needs to run on all localities this main is called on
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    return hpx::init(desc_commandline, argc, argv, cfg);
}
