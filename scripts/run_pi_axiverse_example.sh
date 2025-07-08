#!/bin/bash
#SBATCH --job-name pi_axiverse_example
#SBATCH --output ./logs/%x/log-%j.txt
#SBATCH -p GravityTheory                 # GravityTheory, physics, secondary, IllinoisComputes, or test
#SBATCH --nodes 1                        # [3]            [9]      [308]      [22]                 [2]
#SBATCH --cpus-per-task=128              # [128+]         [24+]    [16+]      [128]                [32]
#SBATCH --time 04:00:00                  # [30d]          [7d]     [4h]       [3d]                 [1h]
#SBATCH --mem 0                          # [1031G+]       [193G+]  [193G+]    [515G]               [16G]
#SBATCH --qos normal

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda3
#module load texlive
source activate pyaxiverse
conda info

# Get SLURM job environment variables
INPUT_SYS_NAME="${SLURM_JOB_NAME}" 
PIAXI_SYS_NAME="${INPUT_SYS_NAME}${PIAXI_JOB_SUFFIX}"
PIAXI_N_CORES="${SLURM_JOB_CPUS_PER_NODE}"
PIAXI_N_NODES="${SLURM_JOB_NUM_NODES}"
PIAXI_COREMEM="${SLURM_MEM_PER_NODE}"
PIAXI_JOB_QOS="${SLURM_JOB_QOS}"
CPUS_ON_NODE=$(nproc)
PIAXI_SLURM_ARGS="--num_cores ${PIAXI_N_CORES} --num_nodes ${PIAXI_N_NODES} --job_qos ${PIAXI_JOB_QOS} --mem_per_node ${PIAXI_COREMEM}"
echo "SLURM ARGS: ${PIAXI_SLURM_ARGS}"

#python piaxiverse.py --no-skip_existing --num_cores 128 --tN 100 --use_natural_units --use_mass_units --verbosity 9 --kN 50 --scan_mass -40 -60 --scan_mass_N 10 --scan_Lambda4 8 11 --scan_Lambda4_N 4 --config_name '3_neutrals_test' --F '1e9' --make_plots --save_output_files
#python piaxiverse.py $PIAXI_SLURM_ARGS --config_name piaxiverse_main1_SU3_seeded --seed 26929701748696353794921921256778610131 --int_method BDF --use_logsumexp --use_mass_units --use_natural_units --verbosity 6 --t 20 --tN 1000 --kN 30 --k_res 0.1 --save_output_files --make_plots --no-show_plots --rho 1.000e+40 --F 3.557e+11  --m_scale 5.240e-33 --L3 1.000e+00 --L4 1.000e+15 --eps 1.000e+00 --dqm_c x x x 0 0 0  --no-skip_existing
python piaxiverse.py $PIAXI_SLURM_ARGS --config_name piaxiverse_main1_SU3_seeded --seed 190880646500800584517504435729576304562 --int_method BDF --use_logsumexp --use_mass_units --use_natural_units --verbosity 8 --t 200 --tN 1000 --kN 100 --k_res 0.1 --config_name piaxiverse_main1_SU3 --save_output_files --make_plots --no-show_plots --rho 1.000e+00 --F 4.743e+14  --m_scale 1.054e-32 --L3 1.000e+00 --L4 1.000e+00 --eps 1.000e+00 --dqm_c x x x 0 0 0  --no-skip_existing
