# High Performance Conjugate Gradient

## Build

```bash
# Make sure to have modules/paths setup correctly
# Internally uses git and Makefile
# HAICGU example        Architecture  Path to Linear Algebra lib
./download_and_build.sh Kunpeng920    /software/kp920-RL8/Stages/2024/software/OpenBLAS/0.3.27-GCC-14.1.0/lib/libopenblas.a
```

## Notes

- The `gen_dat_file.sh` script requires the `sympy` Python package
- The `gen_dat_file.sh` script requires the `awk` command


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

1) Sync the `hpl_results_<cluster>.csv` files on one machine
2) Run:
```bash
python3 plots.py hpl_results_<cluster1>.csv hpl_results_<cluster2>.csv
```