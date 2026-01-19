# Custom Graph500 (BFS-only)

## Build

```bash
# Make sure to have modules/paths setup correctly
# Internally uses Makefile
./compile.sh
```

## Run

```bash
# Assuming you set up SbatchMan aliases 
sbmi
sbmc -f configs.yaml -ow
sbml -f jobs.yaml
```

## Run on Leonardo

This script ensures that the nodes for the experiments are on different L1 switches but under the same L2 switch (same cell, different switch). Nodes are divided into two equally sized groups each under a different L1 switch.

```bash
# Run from 'HPC-benchmarks/Graph500/'
python run_on_leonardo.py --csv ../machines/Leonardo/leo_map.txt --jobs jobs.yaml
```

## Generate Data + (local) Plots

Once all experiments are done:

```bash
python3 plots.py
```

## Plots with data from Multiple Sources

1) Sync the `graph500_<cluster>_data.parquet` files on one machine
2) Run:
```bash
python3 plots.py path/to/graph500_<cluster1>_data.parquet path/to/graph500_<cluster2>_data.parquet ...
```