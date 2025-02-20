#!/bin/bash
#SBATCH --job-name pi_axiverse_example
#SBATCH --output ./logs/%x/log-%j-%a.txt
#SBATCH -p GravityTheory                 # GravityTheory, physics, secondary, IllinoisComputes, or test
#SBATCH --nodes 1                        # [3]            [9]      [308]      [22]                 [2]
#SBATCH --cpus-per-task=128              # [128+]         [24+]    [16+]      [128]                [32]
#SBATCH --time 04:00:00                  # [30d]          [7d]     [4h]       [3d]                 [1h]
#SBATCH --mem 0                          # [1031G+]       [193G+]  [193G+]    [515G]               [16G]
#SBATCH --qos normal

# vvv (Rejoin with the above to have this line count) vvv
#SBATCH --ntasks-per-node=1

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda3
#module load texlive
#source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
#source /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
source activate pyaxiverse
conda info

python piaxiverse.py --no-skip_existing --num_cores 128 --tN 100 --use_natural_units --use_mass_units --verbosity 9 --kN 50 --scan_mass -40 -60 --scan_mass_N 10 --scan_Lambda4 8 11 --scan_Lambda4_N 4 --config_name '3_neutrals_test' --F '1e9' --make_plots --save_output_files
