cmake_minimum_required(VERSION 3.17)

project(allegro-gpu LANGUAGES CXX CUDA)

include_directories("C:\\lib\\allegro-msvc\\Allegro.5.2.8\\build\\native\\include")
link_directories("C:\\lib\\allegro-msvc\\Allegro.5.2.8\\build\\native\\v143\\win32\\lib")

add_executable(a.out main.cpp kernel.cu)
add_executable(a_cpu.out main_cpu.cpp cpu.cpp)

target_link_libraries(a.out allegro allegro_font allegro_primitives)
target_link_libraries(a_cpu.out allegro allegro_font allegro_primitives)