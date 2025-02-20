#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 0:30:00
#SBATCH --mem 4G
#SBATCH --job-name pi_axiverse_test
#SBATCH --output ./logs/%x/log-test-%J.txt
#SBATCH -p test

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=208

module load anaconda3
#module load texlive
#source /gpfs/runtime/opt/anaconda/latest/etc/profile.d/conda.sh
#export PYTHONPATH=/home/${USER}/config/python:${PYTHONPATH}
#. /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
#source /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
source activate pyaxiverse
conda info

python piaxiverse.py --no-skip_existing --num_cores 50 --tN 100 --use_natural_units --use_mass_units --verbosity 9 --kN 50 --scan_mass -40 -60 --scan_mass_N 10 --scan_Lambda4 8 11 --scan_Lambda4_N 4 --config_name '3_neutrals_test' --F '1e9' --make_plots --save_output_files
