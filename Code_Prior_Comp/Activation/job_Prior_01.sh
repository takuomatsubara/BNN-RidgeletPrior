#!/bin/bash
#
#SBATCH -t 01-23:00:00
#SBATCH -c 16
#
python Experiment_Prior_Gauss.py --core 16