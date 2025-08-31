#!/bin/bash

set -e

# HAICGU modules: ml GCC/14.1.0 OpenMPI/5.0.3
# Leonardo modules: ml gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2
# Nanjing: exoprt PATH="/root/hpc/build/gcc/bin:/root/hpc/build/bin:$PATH"; export LD_LIBRARY_PATH="/root/hpc/build/gcc/lib64:/root/hpc/build/lib:$LD_LIBRARY_PATH"

cd graph500/src
mkdir -p ../../bin

# Standard benchmark
make clean
CFLAGS="-DBENCHPIN" make
mv graph500_reference_bfs_sssp ../../bin
mv graph500_reference_bfs ../../bin

# Benchmarks that flushes the buffer more often: after 512 vertices instead of 8192
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=2048 -DAGGR=2048" make
mv graph500_reference_bfs_sssp ../../bin/graph500_reference_bfs_sssp_smallbuf
mv graph500_reference_bfs ../../bin/graph500_reference_bfs_smallbuf

# Benchmarks that flushes the buffer less often: after 512 vertices instead of 8192
make clean
CFLAGS="-DBENCHPIN" PREPROCESSOR_FLAGS="-DAGGR_intra=524288 -DAGGR=524288" make
mv graph500_reference_bfs_sssp ../../bin/graph500_reference_bfs_sssp_largebuf
mv graph500_reference_bfs ../../bin/graph500_reference_bfs_largebuf

echo "WARNING: currently only BFS implements custom metric correctly."