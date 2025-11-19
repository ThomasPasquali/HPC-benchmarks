cmake_minimum_required(VERSION 3.24.3)

project(Popcorn_Kernel_Kmeans VERSION 1.0.0 LANGUAGES CXX)

set(NOFORTRAN ON)
find_package(OpenBLAS REQUIRED)

# find_package(OpenMP)
# if (OPENMP_FOUND)
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

set(CCUTILS_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA support in ccutils")
set(CCUTILS_ENABLE_MPI OFF CACHE BOOL "Enable MPI support in ccutils")
add_subdirectory(ccutils)

message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "OpenBLAS_LIBRARIES: ${OpenBLAS_LIBRARIES}")

add_executable(popcornkmeans
  ./src/kernels.cpp
  ./src/kmeans.cpp
  ./src/main.cpp
)

## flang_rt.runtime may be specific for the HCA cluster 
target_link_libraries(popcornkmeans PRIVATE ${OpenBLAS_LIBRARIES} flang_rt.runtime ccutils::ccutils) # OpenMP::OpenMP_CXX
target_include_directories(popcornkmeans PRIVATE ${OpenBLAS_INCLUDE_DIRS} ./include)
# target_compile_options(popcornkmeans PRIVATE "${OpenMP_CXX_FLAGS}")

# Ensure the runtime linker can find OpenMP / OpenBLAS shared libraries when they are
# installed in non-standard locations: prefer link paths and add $ORIGIN-relative rpaths.
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set_target_properties(popcornkmeans PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
# set_target_properties(popcornkmeans PROPERTIES INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN")

