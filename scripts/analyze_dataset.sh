#!/bin/bash
#SBATCH --job-name piaxi_analysis
#SBATCH --output ./logs/%x/log-%j.txt
#SBATCH -p secondary                     # GravityTheory, physics, secondary, IllinoisComputes, or test
#SBATCH --nodes 1                        # [3]            [9]      [308]      [22]                 [2]
#SBATCH --cpus-per-task=24               # [128+]         [24+]    [16+]      [128]                [32]
#SBATCH --time 08:00:00                  # [30d]          [7d]     [4h]       [3d]                 [1h]
#SBATCH --mem 0                          # [1031G+]       [193G+]  [193G+]    [515G]               [16G]
#SBATCH --qos normal
#SBATCH --ntasks-per-node=1

## set NUMEXPR_MAX_THREADS
#export NUMEXPR_MAX_THREADS=416

#module load anaconda3_cpu
#module load texlive
#source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
#source /usr/local/anaconda/5.2.0/python3/etc/profile.d/conda.sh
#source activate pyaxiverse

TEST=${1:-0}
RAW=${2:-0}

if [ "$TEST" -eq 1 ]; then
    VERBOSITY=3
    echo "Running analysis script in test mode"
    echo "  |  verbosity = ${VERBOSITY}"

    LOG_ROOT="piaxi_main1_copy_3"
    #DATA_ROOT="piaxiverse_main1_SU3_sample_100"
    DATA_ROOT="piaxiverse_main1_SU3"
    ARGF_ROOT="piaxiverse_main1_SU3"

    EXTRA_ARGS=" --copy_clean_results"
    
else

    if [ "$RAW" -eq 1 ]; then
        VERBOSITY=1
        echo "Running analysis script in normal (raw data) mode"
        echo "  |  verbosity = ${VERBOSITY}"

        ./scripts/copy_dataset.sh

        LOG_ROOT="piaxi_main1"
        DATA_ROOT="piaxiverse_main1_SU3"
        ARGF_ROOT="piaxiverse_main1_SU3"
        EXTRA_ARGS=" --copy_clean_results"
        
    else
        VERBOSITY=2
        echo "Running analysis script in normal (clean data) mode"
        echo "  |  verbosity = ${VERBOSITY}"

        LOG_ROOT="piaxiverse_main1_SU3_clean"
        DATA_ROOT="piaxiverse_main1_SU3_clean"
        ARGF_ROOT="piaxiverse_main1_SU3"
        EXTRA_ARGS=""
    fi
fi


LOG_DIR="~/projects/pyAxiverse/logs/${LOG_ROOT}/"
DATA_DIR="~/projects/pyAxiverse/data/${DATA_ROOT}/"
SCRATCH_LOG_DIR="~/scratch/pyAxiverse/logs/scratch/${LOG_ROOT}/"
SCRATCH_DATA_DIR="~/scratch/pyAxiverse/v3.2.8/${DATA_ROOT}/"
ARGFILE_PATH="~/projects/pyAxiverse/ARGFILES/${ARGF_ROOT}"

#rsync -azP "${SCRATCH_DATA_DIR}${DATA_ROOT}_[a-z]*" $DATA_DIR
#rsync -azP "${SCRATCH_DATA_DIR}${DATA_ROOT}_[0-9]*" $DATA_DIR

python "./tools/dataset_utils.py" --log_dir $LOG_DIR --output_dir $DATA_DIR --scratch_dir $SCRATCH_LOG_DIR --argfile_path $ARGFILE_PATH --verbosity $VERBOSITY $EXTRA_ARGS