#!/bin/bash

set -e

ml GCC/14.1.0 OpenMPI/5.0.3

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
