#!/bin/bash
#SBATCH -n 100
#SBATCH -N 1
#SBATCH --time 48:00:00
#SBATCH --mem 200G
#SBATCH --job-name pi_axiverse_array
#SBATCH --output ./logs/pi_axiverse_log-%J.txt
#SBATCH -p batch

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load miniconda3/23.11.0s
module load texlive
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate piaxiverse

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
PIAXI_COREMEM="${SLURM_MEM_PER_NODE}"
PIAXI_JOB_QOS="${SLURM_JOB_QOS}"

PIAXI_SLURM_ARGS="--num_cores ${PIAXI_N_CORES} --num_nodes ${PIAXI_N_NODES} --job_qos ${PIAXI_JOB_QOS} --mem_per_core ${PIAXI_COREMEM}"

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