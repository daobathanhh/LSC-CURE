set(LSC_CURE_SOURCES
    cure/cluster.cpp
    cure/cure.cpp
    cure/distance.cpp
    cure/kd_tree.cpp
)

add_library(lsc_cure ${LSC_CURE_SOURCES})

target_include_directories(lsc_cure
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(lsc_cure PUBLIC cxx_std_17)

add_library(lsc::cure ALIAS lsc_cure)

