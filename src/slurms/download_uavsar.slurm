#!/bin/bash
#SBATCH -J download_uavsar_to_netcdf # job name
#SBATCH -o /bsuhome/zacharykeskinen/uavsar-validation/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 24:00:00 # run time (hh:mm:ss) - 1 minute in this example.

eval "$(conda shell.bash hook)"
conda activate proj

python /bsuhome/zacharykeskinen/uavsar-validation/src/data_acquisition/uavsar_to_nc.py