#!/bin/bash
#SBATCH -J create_netcdfs # job name
#SBATCH -o /bsuhome/zacharykeskinen/uavsar-validation/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bigmem # queue (partition)
#SBATCH -t 8:00:00 # run time (hh:mm:ss) - 1 minute in this example.
#SBATCH --mail-user=zacharykeskinen@boisestate.edu
#SBATCH --mail-type=END

echo "Starting."

eval "$(conda shell.bash hook)"
conda activate validation

echo "Activated environment.."

python -u /bsuhome/zacharykeskinen/uavsar-validation/src/data_acquisition/create/create_netcdfs.py

echo "Done"