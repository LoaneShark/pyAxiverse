#!/bin/bash
#SBATCH -n 100
#SBATCH --time 48:00:00
#SBATCH --mem 5G
#SBATCH --job-name pi_axiverse_star
#SBATCH --output pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=208

module load anaconda/latest
source /gpfs/runtime/opt/anaconda/latest/etc/profile.d/conda.sh
conda activate piaxiverse
if [ $PIAXI_VERBOSITY > 3 ]
then
    conda info
fi

PIAXI_DENSITY="1e22"
PIAXI_SYS_NAME=$SLURM_JOB_NAME
PIAXI_N_CORES=$SLURM_JOB_NUM_NODES
PIAXI_N_TIMES=300
PIAXI_N_KMODE=200
PIAXI_N_QMASS=20
PIAXI_VERBOSITY=5
PIAXI_MASS_UNIT=True
PIAXI_MASS_RANGE="-20 -40"
#PIAXI_L_RANGE="0 30"
#PIAXI_L3_RANGE="0 30"
PIAXI_L4_RANGE="0 30"
#PIAXI_N_L=30
#PIAXI_N_L3=30
PIAXI_N_L4=30

python piaxiverse.py --num_cores $PIAXI_N_CORES --tN $PIAXI_N_TIMES --use_mass_units $PIAXI_MASS_UNIT --verbosity $PIAXI_VERBOSITY --kN $PIAXI_N_KMODE --scan_mass $PIAXI_MASS_RANGE --scan_mass_N $PIAXI_N_QMASS --scan_Lambda4 $PIAXI_L4_RANGE --scan_Lambda4_N $PIAXI_N_L4 --config_name $PIAXI_SYS_NAME --rho $PIAXI_DENSITY --dqm_c 1 1 1 0 0 0 --scan_F 16 18 --scan_F_N 3
