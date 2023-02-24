#!/bin/bash
#SBATCH -J add_insitu_to_uavsar # job name
#SBATCH -o /bsuhome/zacharykeskinen/uavsar-validation/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 6:00:00 # run time (hh:mm:ss) - 1 minute in this example.

eval "$(conda shell.bash hook)"
conda activate proj

python /bsuhome/zacharykeskinen/uavsar-validation/src/comparison/uavsar_swe_change.py

echo "Done"