#!/bin/bash
#SBATCH -J lidar_images # job name
#SBATCH -o /bsuhome/zacharykeskinen/uavsar-validation/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 3:00:00 # run time (hh:mm:ss) - 1 minute in this example.

echo "Starting."

eval "$(conda shell.bash hook)"
conda activate proj

echo "Activated environment.."

python -u /bsuhome/zacharykeskinen/uavsar-validation/src/compare/lidar_images.py

echo "Done"