cmake_minimum_required(VERSION 3.22)

# set project settings
project(TracccStandalone LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# debug flags
# set(CMAKE_CUDA_FLAGS_DEBUG "-G -g") 
# set(CMAKE_CXX_FLAGS_DEBUG "-g") 

include(FetchContent)

# Include CUDA headers
include_directories(/usr/local/cuda/include/)

# find required traccc packages
# find_package(actsvg REQUIRED)
find_package(covfie REQUIRED)
find_package(traccc REQUIRED)
find_package(CUDAToolkit REQUIRED)

message(STATUS "Printing all traccc related variables:")
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    if (_variableName MATCHES "traccc")
        message(STATUS "${_variableName}=${${_variableName}}")
    endif()
endforeach()

message(STATUS "Printing all detray related variables:")
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    if (_variableName MATCHES "detray")
        message(STATUS "${_variableName}=${${_variableName}}")
    endif()
endforeach()

add_executable(TracccGpuStandalone 
    src/TracccGpuStandalone.cpp)

target_include_directories(TracccGpuStandalone 
    PUBLIC src/${traccc_INCLUDE_DIR}
)

add_definitions(-DALGEBRA_PLUGINS_INCLUDE_ARRAY)

target_link_libraries(TracccGpuStandalone PUBLIC 
    vecmem::core vecmem::cuda traccc::io traccc::performance
    traccc::core traccc::device_common traccc::cuda CUDA::cudart
    traccc::options detray::io detray::core)
