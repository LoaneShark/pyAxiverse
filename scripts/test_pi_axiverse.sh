#!/bin/bash
#SBATCH -n 10
#SBATCH --time 1:00:00
#SBATCH --mem 4G
#SBATCH --job-name pi_axiverse
#SBATCH --output pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=208

module load anaconda/latest
source /gpfs/runtime/opt/anaconda/latest/etc/profile.d/conda.sh
conda activate piaxiverse
conda info

python piaxiverse.py --num_cores 10 --tN 50 --use_mass_units True --verbosity 9 --kN 50 --scan_mass -20 -40 --scan_mass_N 5 --scan_Lambda4 10 30 --scan_Lambda4_N 5 --config_name 3_neutrals_test --dqm_c 1 1 1 0 0 0 --scan_F 16 18 --scan_F_N 3
