# HPC-benchmarks

This repository gathers HPC benchmarks:
- Source code or a script to get it
- SbatchMan Configurations and Jobs YAML files
- Scripts for parsing results and plotting

## List of Shared-Memory Benchmarks

- STREAM
- Pointer Chasing
- ThreadsSynchronization

## List of Distributed-Memory Benchmarks (estimated runtime)

- Graph500 (HAICGU)
    - 2 nodes, scale 20, edgefactor 64 , ~4min
    - 4 nodes, scale 20, edgefactor 64, ~2min
- HPL (2 hours)
- HPCG (HAICGU)
    - 2 nodes, 128x128x128 grid, min runtime 30s , ~5min
- DNNProxies (15 min)

*Estimated runtimes refer to experiments up to 8 nodes on a single partition*


Please refer to `README` of the individual folders for more details (WIP)