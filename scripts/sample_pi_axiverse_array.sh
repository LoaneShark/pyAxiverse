#!/bin/bash
#SBATCH -a 1-1000%100      # max: 22680
#SBATCH --job-name pi_axiverse_array
#SBATCH --output ./logs/%x/log-%J-%a.txt
#SBATCH -p secondary                     # GravityTheory, physics, secondary, IllinoisComputes, or test
#SBATCH --nodes 1                        # [3]            [9]      [308]      [22]                 [2]
#SBATCH --cpus-per-task=24               # [128+]         [24+]    [16+]      [128]                [32]
#SBATCH --time 04:00:00                  # [30d]          [7d]     [4h]       [3d]                 [1h]
#SBATCH --mem 0                          # [1031G+]       [193G+]  [193G+]    [515G]               [16G]
#SBATCH --ntasks-per-node=1

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

module load anaconda3
module load texlive
#source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
source /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
conda activate pyaxiverse

INPUT_ARGFILE="${1}"
#INPUT_STARTLINE="${2:-1}"
#INPUT_ENDLINE="${3:-100}"
INPUT_VERBOSITY="${2:-1}"
INPUT_OFFSET=${3}
INPUT_NUM=${4:1000}

# Use OFFSET arg to determine starting value and iterate through for this chunk
# (Workaround because SLURM MaxArraySize is 1000)
JOB_ARRAY_START=${INPUT_OFFSET+1}
JOB_ARRAY_END=${INPUT_OFFSET+INPUT_NUM}
JOB_ARRAY_VALUES=({${JOB_ARRAY_START}..${JOB_ARRAY_END}})
JOB_ARRAY_INDEX=${JOB_ARRAY_VALUES[$SLURM_ARRAY_TASK_ID]}

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

PIAXI_SLURM_ARGS="--num_cores '${PIAXI_N_CORES}' --num_nodes ${PIAXI_N_NODES} --job_qos ${PIAXI_JOB_QOS} --job_partition ${PIAXI_JOB_PARTITION} ${PIAXI_MEM_ARG}"

if [[ "${PIAXI_VERBOSITY}" -gt "0" ]]
then
    #echo "RANGE: ${INPUT_STARTLINE} ${INPUT_ENDLINE}"
    echo "ARGFILE: ${INPUT_ARGFILE}"
fi

if [[ "${PIAXI_VERBOSITY}" -gt "6" ]]
then
    echo "PIAXI_SLURM_ARGS: ${PIAXI_SLURM_ARGS}"
fi

#for i in $(seq $INPUT_STARTLINE $INPUT_ENDLINE)
#do
PIAXI_COMMAND=$(sed -n "${JOB_ARRAY_INDEX}p" < ${INPUT_ARGFILE})
#PIAXI_COMMAND=$(sed -n "${i}p" < ${INPUT_ARGFILE})
echo "LINE ${i}:  ${PIAXI_COMMAND}"

eval "${PIAXI_COMMAND} ${PIAXI_SLURM_ARGS}"
#done