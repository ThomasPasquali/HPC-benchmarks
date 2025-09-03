# High Performance Conjugate Gradient

## Build

```bash
# Make sure to have modules/paths setup correctly
# Internally uses git and CMake
./download_and_build.sh
```

## Run

```bash
# Assuming you set up SbatchMan aliases 
sbmi
sbmc -f configs.yaml -ow
sbml -f jobs.yaml
```

## Generate Data

Once all experiments are done:

```bash
python3 parse_results.py
```

## Plots with data from Multiple Sources

1) Sync the `hpcg_results_<cluster>.csv` files on one machine
2) Run:
```bash
python3 plots.py hpcg_results_<cluster1>.csv hpcg_results_<cluster2>.csv ...
```