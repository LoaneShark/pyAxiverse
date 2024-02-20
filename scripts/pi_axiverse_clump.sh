#!/bin/bash
#SBATCH -n 100
#SBATCH --time 48:00:00
#SBATCH --mem 10G
#SBATCH --job-name pi_axiverse_clump
#SBATCH --output pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda
module load texlive
source /gpfs/runtime/opt/anaconda/latest/etc/profile.d/conda.sh
conda activate piaxiverse
if [[ $PIAXI_VERBOSITY > 3 ]]
then
    conda info
fi

INPUT_ARG1="${1:-"0"}"
PIAXI_JOB_SUFFIX=""
if [[ "$INPUT_ARG1" = "0" ]]
then
    PIAXI_DQMC="1 1 1 0 0 0"
elif [[ "$INPUT_ARG1" = "SIMPLE" ]]
then
    PIAXI_DQMC="1 1 1 0 0 0"
    PIAXI_JOB_SUFFIX="_simple"
elif [[ "$INPUT_ARG1" = "FULL" ]]
then
    PIAXI_DQMC="1 1 1 1 1 1"
    PIAXI_JOB_SUFFIX="_full"
elif [[ "$INPUT_ARG1" = "SAMPLED" ]]
then
    PIAXI_DQMC="None None None None None None"
    PIAXI_JOB_SUFFIX="_sampled"
fi

PIAXI_SYS_NAME="${SLURM_JOB_NAME}${PIAXI_JOB_SUFFIX}"
PIAXI_N_CORES=$SLURM_JOB_NUM_NODES
PIAXI_COREMEM=$SLURM_MEM_PER_NODE

PIAXI_DENSITY="1e5"
PIAXI_VERBOSITY=5
PIAXI_N_SAMPLES=3

PIAXI_N_TIMES=300
PIAXI_MAX_TIME=30
PIAXI_MAX_KMODE=200
PIAXI_KMODE_RES=0.1

PIAXI_N_QMASS=5
PIAXI_MASS_RANGE="-60 -80"

#PIAXI_L_RANGE="0 30"
#PIAXI_N_L=30
if [[ "$INPUT_ARG1" = "FULL" ]]
then
    PIAXI_L3_RANGE="0 30"
    PIAXI_N_L3=5
else
    PIAXI_L3_RANGE="None"
    PIAXI_N_L3="None"
fi
PIAXI_L4_RANGE="0 30"
PIAXI_N_L4=5

#PIAXI_F="1e9"
PIAXI_F_RANGE="5 10"
PIAXI_N_F=5

PIAXI_N_EPS=5
PIAXI_EPS_RANGE="-25 -5"

python piaxiverse.py --use_natural_units --use_mass_units --num_cores $PIAXI_N_CORES --mem_per_core $PIAXI_COREMEM --num_samples $PIAXI_N_SAMPLES --t $PIAXI_MAX_TIME --tN $PIAXI_N_TIMES --use_mass_units $PIAXI_MASS_UNIT --verbosity $PIAXI_VERBOSITY --k $PIAXI_MAX_KMODE --k_res $PIAXI_KMODE_RES --scan_mass $PIAXI_MASS_RANGE --scan_mass_N $PIAXI_N_QMASS --scan_Lambda4 $PIAXI_L4_RANGE --scan_Lambda4_N $PIAXI_N_L4 --config_name $PIAXI_SYS_NAME --rho $PIAXI_DENSITY --dqm_c $PIAXI_DQMC --scan_F $PIAXI_F_RANGE --scan_F_N $PIAXI_N_F --scan_epsilon $PIAXI_EPS_RANGE --scan_epsilon_N $PIAXI_N_EPS --save_output_files --make_plots
