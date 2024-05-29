#!/bin/bash
#SBATCH -n 100
#SBATCH -N 1
#SBATCH --time 48:00:00
#SBATCH --mem 300G
#SBATCH --job-name pi_axiverse_star
#SBATCH --output pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load miniconda3/23.11.0s
module load texlive
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate piaxiverse


#----------------------------------------------------------------------------
#    INPUTS AND SETTINGS
#----------------------------------------------------------------------------

# General Settings
INPUT_VERBOSITY="7"           # Output verbosity [1-10], -1 to disable all output
INPUT_METHOD="BDF"            # Numerical integration method ("RK45" or "BDF")
INPUT_M_UNITS="True"          # Use natural units in calculations (True or False)
INPUT_SEED="None"             # RNG seed (or "None" for random sampling)
INPUT_SCENARIO="SIMPLE"       # Determine which subset of theory space to consider
INPUT_SAVE="True"             # Whether or not to save output files
INPUT_PLOTS="True"            # Whether or not to generate and save plots
INPUT_LOGSCALE="True"         # Whether or not calculations are performed in log-scale

# Simulation Granularity
PIAXI_N_TIMES=300             # Number of timesteps
PIAXI_T_MAX=30                # Max time value
PIAXI_N_KMODE=200             # Number of k_modes
PIAXI_K_RES=1                 # k_mode step-size

# Initial Conditions
INPUT_DENSITY="1e20"          # Local DM energy density [GeV/cm^3]



#----------------------------------------------------------------------------
#    ARGUMENT PARSING LOGIC
#----------------------------------------------------------------------------

PIAXI_VERBOSITY="${INPUT_VERBOSITY}"
if [[ "${PIAXI_VERBOSITY}" -gt "6" ]]
then
    conda info
fi

SCAN_ALL=1
if (( $SCAN_ALL ))
then
    N_SAMPLES=3
else
    N_SAMPLES=1
fi

# Toggle whether calculations are performed in natural units or physical units
if [[ "${INPUT_M_UNITS}" == "False" ]]
then
    PIAXI_UNITS="--no-use_mass_units --no-use_natural_units"
else
    PIAXI_UNITS="--use_mass_units --use_natural_units"
fi

# RNG seed
if [[ "${INPUT_SEED}" == "None" ]]
then
    PIAXI_SEED=""
else
    PIAXI_SEED="--seed ${INPUT_SEED}"
fi

# Determine number of surviving pi-axion species and parameter distributions
PIAXI_JOB_SUFFIX=""
PIAXI_FIT=""
PIAXI_MASKS=""
if [[ "${INPUT_SCENARIO}" == "0" ]]
then
    PIAXI_DQMC="1 1 1 0 0 0"
elif [[ "${INPUT_SCENARIO}" == "SINGLE" ]]
then
    PIAXI_DQMC="0.5 0.5 0 0 0 0"
    PIAXI_JOB_SUFFIX="_single"
elif [[ "${INPUT_SCENARIO}" == "QCD" ]] || [[ "${INPUT_SCENARIO}" == "AXION" ]]
then
    PIAXI_DQMC="0.5 0.5 0 0 0 0"
    PIAXI_JOB_SUFFIX="_qcd"
    PIAXI_FIT="--fit_QCD"
    PIAXI_MASKS="--mask_complex --mask_charged"
elif [[ "${INPUT_SCENARIO}" == "SIMPLE" ]]
then
    PIAXI_DQMC="1 1 1 0 0 0"
    PIAXI_JOB_SUFFIX="_simple"
elif [[ "${INPUT_SCENARIO}" == "FULL" ]]
then
    PIAXI_DQMC="1 1 1 1 1 1"
    PIAXI_JOB_SUFFIX="_full"
elif [[ "${INPUT_SCENARIO}" == "SU3" ]]
then
    PIAXI_DQMC="x x x 0 0 0"
    PIAXI_JOB_SUFFIX="_SU3"
elif [[ "${INPUT_SCENARIO}" == "SAMPLED" ]] || [[ "${INPUT_SCENARIO}" == "SU6" ]]
then
    PIAXI_DQMC="x x x x x x"
    PIAXI_JOB_SUFFIX="_SU6"
elif [[ "${INPUT_SCENARIO}" == "REALS" ]] || [[ "${INPUT_SCENARIO}" == "AXIVERSE" ]]
then
    PIAXI_DQMC="x x x x x x"
    PIAXI_JOB_SUFFIX="_reals"
    PIAXI_MASKS="--mask_complex --mask_charged"
elif [[ "${INPUT_SCENARIO}" == "NEUTRALS" ]]
then
    PIAXI_DQMC="x x x x x x"
    PIAXI_JOB_SUFFIX="_neutrals"
    PIAXI_MASKS="--mask_charged"
elif [[ "${INPUT_SCENARIO}" == "COMPLEX" ]]
then
    PIAXI_DQMC="x x x x x x"
    PIAXI_JOB_SUFFIX="_complex"
    PIAXI_MASKS="--mask_reals"
elif [[ "${INPUT_SCENARIO}" == "COMPLEX_NEUTRALS" ]]
then
    PIAXI_DQMC="x x x x x x"
    PIAXI_JOB_SUFFIX="_complex_neutrals"
    PIAXI_MASKS="--mask_reals --mask_charged"
elif [[ "${INPUT_SCENARIO}" == "CHARGED" ]]
then
    PIAXI_DQMC="x x x x x x"
    PIAXI_JOB_SUFFIX="_charged"
    PIAXI_MASKS="--mask_reals --mask_complex"
fi

# Whether or not to save output files and plots
if [[ "${INPUT_PLOTS}" == "True" ]]
then
    PIAXI_PLOTS="--make_plots --no-show_plots"
else
    PIAXI_PLOTS="--no-make_plots --no-show_plots"
fi

if [[ "${INPUT_SAVE}" == "True" ]]
then
    PIAXI_SAVE="--save_output_files ${PIAXI_PLOTS}"
else
    PIAXI_SAVE="--no-save_output_files --no-make_plots --no-show_plots"
fi

# Numerical integration method ("RK45" or "BDF")
PIAXI_METHOD="${INPUT_METHOD}"

# Whether or not to perform numerical integration calculations in log-scale
if [[ "${INPUT_LOGSCALE}" == "True" ]]
then
    PIAXI_LOGSCALE="--use_logsumexp"
else
    PIAXI_LOGSCALE=""
fi



#----------------------------------------------------------------------------
#    SLURM ENVIRONMENT CONFIGURATION
#----------------------------------------------------------------------------
INPUT_SYS_NAME="${SLURM_JOB_NAME}" 
PIAXI_SYS_NAME="${INPUT_SYS_NAME}${PIAXI_JOB_SUFFIX}"
PIAXI_N_CORES="${SLURM_JOB_CPUS_PER_NODE}"
PIAXI_N_NODES="${SLURM_JOB_NUM_NODES}"
PIAXI_COREMEM="${SLURM_MEM_PER_NODE}"
PIAXI_JOB_QOS="${SLURM_JOB_QOS}"



#----------------------------------------------------------------------------
#    PARAMETER SPACE SAMPLING RANGES
#----------------------------------------------------------------------------

# Density [GeV] || For QCD axion case ~ (amp_a)^2*m_a / 2
if (( $SCAN_ALL ))
then
    PIAXI_DENSITY_RANGE="20 30"
    PIAXI_DENSITY_N=5
    PIAXI_DENSITY_ARGS="--scan_rho ${PIAXI_DENSITY_RANGE} --scan_rho_N ${PIAXI_DENSITY_N}"
else
    PIAXI_DENSITY="${INPUT_DENSITY}"
    PIAXI_DENSITY_ARGS="--rho ${PIAXI_DENSITY}"
fi

# F_pi [GeV] || For QCD axion case ~ 2/g_a
if (( $SCAN_ALL ))
then
    PIAXI_F_RANGE="10 50"
    PIAXI_N_F=5
    PIAXI_F_ARGS="--scan_F ${PIAXI_F_RANGE} --scan_F_N ${PIAXI_N_F} ${PIAXI_FIT}"
else
    PIAXI_F="1e10"
    PIAXI_F_ARGS="--F ${PIAXI_F} ${PIAXI_FIT}"
fi

# m_I [eV] || For QCD axion case ~ (m_a)^2 / F_pi
if (( $SCAN_ALL ))
then
    PIAXI_N_QMASS=10
    PIAXI_MASS_RANGE="-80 -20"
    PIAXI_M_ARGS="--scan_mass ${PIAXI_MASS_RANGE} --scan_mass_N ${PIAXI_N_QMASS}"
else
    PIAXI_MASS="1e-80"
    PIAXI_M_ARGS="--m_scale ${PIAXI_MASS}"
fi

# Pi-Axiverse coupling constants: Lambda_3 and Lambda_4 [GeV]
if (( $SCAN_ALL ))
then
    if [[ "${INPUT_SCENARIO}" == "FULL" ]] || [[ "${INPUT_SCENARIO}" == "SAMPLED" ]] || [[ "${INPUT_SCENARIO}" == "SU6" ]] || [[ "${INPUT_SCENARIO}" == "CHARGED" ]] || [[ "${INPUT_SCENARIO}" == "COMPLEX" ]]
    then
        PIAXI_L3_RANGE="10 50"
        PIAXI_N_L3=5
        L3_ARGS="--scan_Lambda3 ${PIAXI_L3_RANGE} --scan_Lambda3_N ${PIAXI_N_L3}"
    else
        PIAXI_L3_RANGE="None"
        PIAXI_N_L3="None"
        L3_ARGS=""
    fi
    PIAXI_L4_RANGE="10 50"
    PIAXI_N_L4=5
    L4_ARGS="--scan_Lambda4 ${PIAXI_L4_RANGE} --scan_Lambda4_N ${PIAXI_N_L4}"
    PIAXI_L_ARGS="${L3_ARGS} ${L4_ARGS}"
else
    PIAXI_L3="1e0"
    PIAXI_L4="1e10"
    PIAXI_L_ARGS="--L3 ${PIAXI_L3} --L4 ${PIAXI_L4}"
fi

# Millicharge: epsilon
if (( $SCAN_ALL ))
then
    PIAXI_N_EPS=10
    PIAXI_EPS_RANGE="0 -20"
    PIAXI_EPS_ARGS="--scan_epsilon ${PIAXI_EPS_RANGE} --scan_epsilon_N ${PIAXI_N_EPS}"
else
    PIAXI_EPS="1e0"
    PIAXI_EPS_ARGS="--eps ${PIAXI_EPS}"
fi



#----------------------------------------------------------------------------
#    PI-AXIVERSE
#----------------------------------------------------------------------------

# Gather relevant args to pass on to script
PIAXI_RES_ARGS="--t ${PIAXI_T_MAX} --tN ${PIAXI_N_TIMES} --kN ${PIAXI_N_KMODE} --k_res ${PIAXI_K_RES}"
PIAXI_INPUT_ARGS="${PIAXI_SEED} --int_method ${PIAXI_METHOD} ${PIAXI_UNITS} ${PIAXI_LOGSCALE} --num_samples ${N_SAMPLES} --verbosity ${PIAXI_VERBOSITY} ${PIAXI_RES_ARGS} --config_name ${PIAXI_SYS_NAME} ${PIAXI_SAVE}"
PIAXI_PARAM_ARGS="${PIAXI_DENSITY_ARGS} ${PIAXI_F_ARGS} ${PIAXI_M_ARGS} ${PIAXI_L_ARGS} ${PIAXI_EPS_ARGS} --dqm_c ${PIAXI_DQMC} ${PIAXI_MASKS}"
PIAXI_SLURM_ARGS="--num_cores ${PIAXI_N_CORES} --num_nodes ${PIAXI_N_NODES} --job_qos ${PIAXI_JOB_QOS} --mem_per_core ${PIAXI_COREMEM}"

if [[ "${PIAXI_VERBOSITY}" -gt "3" ]]
then
    echo "INPUT ARGS: ${PIAXI_INPUT_ARGS}"
    echo "PARAM ARGS: ${PIAXI_PARAM_ARGS}"
    echo "SLURM ARGS: ${PIAXI_SLURM_ARGS}"
fi

# Run Pi-Axiverse simulation
python piaxiverse.py $PIAXI_INPUT_ARGS $PIAXI_SLURM_ARGS $PIAXI_PARAM_ARGS --skip_existing