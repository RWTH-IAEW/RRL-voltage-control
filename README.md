# Partially Observable Residual Reinforcement Learning for PV-Inverter-Based Voltage Control in Distribution Grids

***

Code accompanying the paper **"Partially Observable Residual Reinforcement Learning for PV-Inverter-Based Voltage Control in Distribution Grids"** by Sarra Bouchkati, Ramil Sabirov, Steffen Kortmann, and Andreas Ulbig.

## Installation
We use [Conda]((https://conda.io/projects/conda/en/latest/user-guide/install/index.html)) to create a Python environment in which the code can be executed.

Run:
````
conda env create --file environment.yml
conda activate voltage-control-env
````

For the experiments we use a newly developed voltage control environment that is published seperately and can also be found on [GitHub](https://github.com/RWTH-IAEW/voltage-control-env). When installing the conda environment the environment should automatically be installed as a package.

## Usage
The training for RL and Residual RL can be started by executing the ```main.py``` script in the 
```src``` directory, e.g. to reproduce our best results from the paper you can run

````
cd src
python main.py -algo "residual" -lam-res 1.5 -qnet-arc "cnn" -ctrl-inc
````

For information about the arguments you can pass to the script, you can simply run
````
python main.py --help
````
to display a help page explaining the usage.

## Logging
The logging of the runs is done with [Weights and Biases](https://wandb.ai). It is necessary to create an account and perform an intial login as described on their website.