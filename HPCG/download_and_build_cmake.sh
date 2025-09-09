#!/bin/bash

set -e

mkdir -p hpcg-out

BRANCHNAME=master
HPCG_PATH=${1:-'hpcg-cpu'}
cxx_compiler=g++
additional_cmake_config=

if [[ ! -d ${HPCG_PATH} ]]; then
  git clone -b $BRANCHNAME --recursive https://github.com/hpcg-benchmark/hpcg.git ${HPCG_PATH}
else
  echo "Directory ${HPCG_PATH} already exists; assuming source code has been downloaded before"
fi

cd ${HPCG_PATH}

# CMake build
if [ -d build ]; then
  echo "directory \"build\" already exists - deleting previous HPCG build"
  rm -rf build;
fi

mkdir build; cd build;
cmake -DCMAKE_CXX_COMPILER=${cxx_compiler} -DCMAKE_BUILD_TYPE=Release -DHPCG_ENABLE_MPI=ON -DHPCG_ENABLE_OPENMP=ON ${additional_cmake_config} ../
make -j 16;

cp ../../hpcg.dat .

echo "Done!"