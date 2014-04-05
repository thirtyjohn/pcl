cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(gen_model)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (gen_model gen_model.cpp)
target_link_libraries (gen_model ${PCL_LIBRARIES})



