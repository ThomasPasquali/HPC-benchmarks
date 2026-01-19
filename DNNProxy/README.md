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
sbl -f jobs_nl.yaml                       # sbl -> sbatchman launch
```

## Generate Data + (local) Plots

Once all experiments are done:

```bash
python3 parser.py
```