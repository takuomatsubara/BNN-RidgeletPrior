#!/bin/bash
#
#SBATCH -t 01-23:00:00
#SBATCH -c 32
#
python Experiment_Prior.py --core 32 --SW 1 --SB 2