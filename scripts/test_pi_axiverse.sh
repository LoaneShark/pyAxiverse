#!/bin/bash
#SBATCH -n 20
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

python piaxiverse.py --no-skip_existing --num_cores 20 --tN 100 --use_natural_units --use_mass_units --verbosity 9 --kN 50 --scan_mass -60 -80 --scan_mass_N 10 --scan_Lambda4 8 11 --scan_Lambda4_N 4 --config_name '3_neutrals_test' --F "1e9" --make_plots --save_output_files
