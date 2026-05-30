cmake_minimum_required(VERSION 3.10)

project(lsc_cure
    VERSION 1.0.0
    LANGUAGES CXX
)

option(LSC_CURE_BUILD_EXAMPLES "Build LSC-CURE command line tools" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(src)

if(LSC_CURE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

