# Distributed Deep Neural Network Proxies (Training)

## Load Modules

### On Leonardo
```bash
# to compile and run the experiments
module load gcc/12.2.0
module load openmpi/4.1.6--gcc--12.2.0-cuda-12.2
# to plot the data and run the "run_on_leonardo.py" script  
# Note: The default python on Leonardo is 3.6.8 (old)
module load python/3.11.7 
```

## Build 
```bash
# After Loading the necessary modules
make
```

## Install Sbatchman

You need Sbatchman to run the experiments. Follow the [development installation guide](https://sbatchman.readthedocs.io/en/latest/development/). 

IMPORTANT: you need to install SbatchMan as a developer! Do NOT follow the standard installation.

## Setup Sbatchman
```bash
# Assuming you’ve set up the SbatchMan aliases:
sbmi                                   # sbmi -> sbatchman init
sbmc -f configs.yaml -ow               # sbmc -> sbatchman configure
```

## Run

```bash
# Assuming you’ve set up the SbatchMan aliases:
sbl -f jobs.yaml                       # sbl -> sbatchman launch
```

## Run on Leonardo
The Python script **run_on_leonardo** ensures that the nodes for the experiments are on different L1 switches but under the same L2 switch (same cell, different switch). Nodes are divided into two equally sized groups each under a different L1 switch.
```bash
python run_on_leonardo.py --csv ../machines/Leonardo/leo_map.txt --jobs jobs.yaml
```

## Generate Data + (local) Plots

Once all experiments are done:

```bash
python3 plots.py
```

## Plots with data from Multiple Sources

1) Sync the `results/dnnproxies_<cluster>_data.csv` files on one machine
2) Run:
```bash
python3 plots.py path/to/dnnproxies_<cluster1>_data.csv path/to/dnnproxies_<cluster2>_data.csv ...
```