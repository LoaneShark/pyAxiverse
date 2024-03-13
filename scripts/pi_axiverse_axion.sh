#!/bin/bash
#SBATCH -n 100
#SBATCH --time 24:00:00
#SBATCH --mem 10G
#SBATCH --job-name pi_axiverse_axion
#SBATCH --output pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda
module load texlive
source /gpfs/runtime/opt/anaconda/latest/etc/profile.d/conda.sh
conda activate piaxiverse

PIAXI_VERBOSITY=8

if [[ $PIAXI_VERBOSITY > 3 ]]
then
    conda info
fi

PIAXI_SYS_NAME=$SLURM_JOB_NAME
PIAXI_N_CORES=$SLURM_JOB_NUM_NODES
PIAXI_COREMEM=$SLURM_MEM_PER_NODE

PIAXI_N_TIMES=300
PIAXI_MAX_TIME=30
PIAXI_MAX_KMODE=200
PIAXI_KMODE_RES=0.1

# Expect resonance for cases where [sqrt(2*Rho)/m_a] >= [F_pi]
# let m_a = 1e-6 eV
# let g_a = 1e-15 GeV^-1
# Expect critical threshold between rho ~ 1e17 GeV and 1e18 GeV

# g_a ~ l4 / (L4^2 * 2)
PIAXI_L4="7.2"

# Density [GeV] ~ (amp_a)^2*m_a / 2
PIAXI_DENSITY="1e22"
# F_pi [GeV] ~ 2/g_a
PIAXI_F="15.2"
# m_I [eV] ~ (m_a)^2 / F_pi
PIAXI_MASS="-37.5"

python piaxiverse.py --use_natural_units --use_mass_units --num_cores $PIAXI_N_CORES --mem_per_core $PIAXI_COREMEM --num_samples 1 --t $PIAXI_MAX_TIME --tN $PIAXI_N_TIMES --use_mass_units $PIAXI_MASS_UNIT --verbosity $PIAXI_VERBOSITY --k $PIAXI_MAX_KMODE --k_res $PIAXI_KMODE_RES --m_scale $PIAXI_MASS --config_name $PIAXI_SYS_NAME --rho $PIAXI_DENSITY --eps=1 --dqm_c 0.5 0.5 0 0 0 0 --no-fit_F --F $PIAXI_F --L4 $PIAXI_L4 --save_output_files --make_plots
