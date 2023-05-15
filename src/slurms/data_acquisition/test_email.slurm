#!/bin/bash
#SBATCH -J test_email # job name
#SBATCH -o /bsuhome/zacharykeskinen/uavsar-validation/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 00:01:00 # run time (hh:mm:ss) - 1 minute in this example.
#SBATCH --mail-user=zacharykeskinen@boisestate.edu
#SBATCH --mail-type=END

echo $PATH
