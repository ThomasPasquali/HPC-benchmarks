#!/bin/bash

set -e

# HAICGU modules: ml GCC/14.1.0 OpenMPI/5.0.3
# Leonardo modules: ml gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 cmake/3.27.9
# Nanjing: exoprt PATH="/root/hpc/build/gcc/bin:/root/hpc/build/bin:$PATH"; export LD_LIBRARY_PATH="/root/hpc/build/gcc/lib64:/root/hpc/build/lib:$LD_LIBRARY_PATH"

cd graph500
if [[ ! -L ccutils ]]; then
    ln -s ../../ccutils/ .
fi
cd src
mkdir -p ../../bin

# Standard benchmark
make deep_clean
CFLAGS="-DBENCHPIN" make graph500_reference_bfs
mv graph500_reference_bfs ../../bin/graph500_bfs_32KiB

# Benchmarks that flushes the buffer more often: after 512 vertices instead of 8192
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=2048 -DAGGR=2048" make graph500_reference_bfs
mv graph500_reference_bfs ../../bin/graph500_bfs_2KiB

# Benchmarks that flushes the buffer less often
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=524288 -DAGGR=524288" make graph500_reference_bfs
mv graph500_reference_bfs ../../bin/graph500_bfs_512KiB

# Benchmarks that flushes the buffer even less often
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=2097152 -DAGGR=2097152" make graph500_reference_bfs
mv graph500_reference_bfs ../../bin/graph500_bfs_2MiB

echo "WARNING: currently only BFS implements custom metric correctly."