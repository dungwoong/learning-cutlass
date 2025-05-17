cmake_minimum_required(VERSION 3.22.1)

project(CUDA_AMPERE LANGUAGES CXX CUDA) # specify project uses C++ and CUDA

# fetch the CUDA library
find_package(CUDA REQUIRED)

# Make sure CUDA language is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 120)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src) # includes src in search path

# store all files under src in SRC variable
# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

add_executable(main main.cu ${SRC})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/cutlass/include)
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cuda)