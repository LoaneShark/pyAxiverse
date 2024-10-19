#!/bin/bash
#SBATCH --nodes 64
#SBATCH --time 04:00:00
#SBATCH --mem 100G
#SBATCH --job-name pi_axiverse_array
#SBATCH --output ./logs/pi_axiverse_log-%J.txt
#SBATCH -p secondary                   # GravityTheory, physics, secondary, or test
#SBATCH --ntasks-per-node=1            # Number of tasks per node (1 per node for parallel execution)


## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda3
module load texlive
source /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate pyaxiverse

INPUT_ARGFILE="${1}"
INPUT_STARTLINE="${2:-1}"
INPUT_ENDLINE="${3:-100}"
INPUT_VERBOSITY="${4:-1}"

# Verbose printouts for debugging conda environment
PIAXI_VERBOSITY="${INPUT_VERBOSITY}"
if [[ "${PIAXI_VERBOSITY}" -gt "6" ]]
then
    conda info
fi

# Performance variables and SLURM environment configuration
PIAXI_N_CORES="${SLURM_JOB_CPUS_PER_NODE}"
PIAXI_N_NODES="${SLURM_JOB_NUM_NODES}"
PIAXI_NODEMEM="${SLURM_MEM_PER_NODE}"
PIAXI_JOB_QOS="${SLURM_JOB_QOS}"
PIAXI_JOB_PARTITION="${SLURM_JOB_PARTITION}"

if [[ -z "${PIAXI_NODEMEM}" ]]
then
    PIAXI_MEM_ARG=""
else
    PIAXI_MEM_ARG="--mem_per_node ${PIAXI_NODEMEM}"
fi

PIAXI_SLURM_ARGS="--num_cores ${PIAXI_N_CORES} --num_nodes ${PIAXI_N_NODES} --job_qos ${PIAXI_JOB_QOS} --job_partition ${PIAXI_JOB_PARTITION} ${PIAXI_MEM_ARG}"

if [[ "${PIAXI_VERBOSITY}" -gt "0" ]]
then
    echo "RANGE: ${INPUT_STARTLINE} ${INPUT_ENDLINE}"
    echo "ARGFILE: ${INPUT_ARGFILE}"
fi

if [[ "${PIAXI_VERBOSITY}" -gt "6" ]]
then
    echo "PIAXI_SLURM_ARGS: ${PIAXI_SLURM_ARGS}"
fi

for i in $(seq $INPUT_STARTLINE $INPUT_ENDLINE)
do
    #PIAXI_COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" < ${INPUT_ARGFILE})
    PIAXI_COMMAND=$(sed -n "${i}p" < ${INPUT_ARGFILE})
    echo "LINE ${i}:  ${PIAXI_COMMAND}"

    eval "${PIAXI_COMMAND} ${PIAXI_SLURM_ARGS}"
done