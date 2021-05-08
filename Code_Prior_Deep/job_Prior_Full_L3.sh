#!/bin/bash
#
#SBATCH -t 01-23:00:00
#SBATCH -c 32
#
python Experiment_Prior_Full_L3.py --core 32