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

## Generate Data + (local) Plots

Once all experiments are done:

```bash
python3 plots.py
```

## Plots with data from Multiple Sources

1) Sync the `graph500_<cluster>_summary_aggr.csv` and `graph500_<cluster>_summary.csv` files on one machine
2) Run:
```bash
python3 plots.py \
    path/to/graph500_<cluster1>_summary_aggr.csv path/to/graph500_<cluster1>_summary.csv \
    path/to/graph500_<cluster2>_summary_aggr.csv path/to/graph500_<cluster2>_summary.csv \
    ...
```