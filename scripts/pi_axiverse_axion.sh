#!/bin/bash
#SBATCH -n 100
#SBATCH --time 48:00:00
#SBATCH --mem 5G
#SBATCH --job-name pi_axiverse_axion
#SBATCH --output pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=208

module load anaconda/latest
source /gpfs/runtime/opt/anaconda/latest/etc/profile.d/conda.sh
conda activate piaxiverse

PIAXI_VERBOSITY=8

if [ $PIAXI_VERBOSITY > 3 ]
then
    conda info
fi

PIAXI_SYS_NAME=$SLURM_JOB_NAME
PIAXI_N_CORES=$SLURM_JOB_NUM_NODES
PIAXI_N_TIMES=300
PIAXI_N_KMODE=400

# Expect resonance for cases where [sqrt(2*Rho)/m_a] >= [F_pi]
# let m_a = 1e-6 eV
# let g_a = 1e-10 GeV^-1
# Expect critical threshold between rho ~ 1e17 GeV and 1e18 GeV

# Density [GeV] ~ (amp_a)^2*m_a / 2
PIAXI_DENSITY="1e17"
# F_pi [GeV] ~ 2/g_a
PIAXI_F="2e10"
# m_I [eV] ~ (m_a)^2 / F_pi
PIAXI_MASS="5e-32"

python piaxiverse.py --use_natural_units True --num_cores $PIAXI_N_CORES --tN $PIAXI_N_TIMES --verbosity $PIAXI_VERBOSITY --kN $PIAXI_N_KMODE --m_scale $PIAXI_MASS --config_name $PIAXI_SYS_NAME --rho $PIAXI_DENSITY --dqm_c 0.5 0.5 0 0 0 0 --fit_F False --F $PIAXI_F
