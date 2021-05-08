#!/bin/bash
#
#SBATCH -t 01-23:00:00
#SBATCH -c 16
#
python Experiment_Prior.py --core 16