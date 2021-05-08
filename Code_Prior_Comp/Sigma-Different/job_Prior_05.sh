#!/bin/bash
#
#SBATCH -t 01-23:00:00
#SBATCH -c 32
#
python Experiment_Prior_Nolabel.py --core 32 --SW 5 --SB 30