#!/bin/bash
#SBATCH -J add_sd_interval # job name
#SBATCH -o /bsuhome/zacharykeskinen/uavsar-validation/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 18:00:00 # run time (hh:mm:ss) - 1 minute in this example.

echo "Starting."

eval "$(conda shell.bash hook)"
conda activate proj

echo "Activated environment.."

python -u /bsuhome/zacharykeskinen/uavsar-validation/src/processing/interval_add_sd.py

echo "Done"