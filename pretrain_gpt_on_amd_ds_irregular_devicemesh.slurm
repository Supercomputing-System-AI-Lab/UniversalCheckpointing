#!/bin/bash
#SBATCH -J pretrain-gpt-debug          # Job name
#SBATCH -o slurm_logs/%x_%j.out      # Name of stdout output file (%j expands to jobId)
#SBATCH -N 4                     # Total number of nodes requested
#SBATCH -t 00:04:00              # Run time (hh:mm:ss)
#SBATCH -p mi2508x


DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo $DATETIME

export EXIT_INTERVAL=5

# Calculate total parallelism
TOTAL_PARALLELISM=$((TP * PP * DP * SP))


########################################################
# start
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "GPUS_PER_NODE_IRREGULAR: $GPUS_PER_NODE_IRREGULAR"
echo "TOTAL_PARALLELISM: $TOTAL_PARALLELISM"
echo "SLURM_NNODES: $SLURM_NNODES"

if [ -z "$GPUS_PER_NODE_IRREGULAR" ]; then
    export GPUS_PER_NODE=$((TOTAL_PARALLELISM / SLURM_NNODES))

    # Validate GPUS_PER_NODE
    if [ "$GPUS_PER_NODE" -le 0 ]; then
        echo "Error: GPUS_PER_NODE must be greater than 0: $GPUS_PER_NODE"
        exit 1
    fi
    export GPUS_PER_NODE_IRREGULAR=""
else
    export GPUS_PER_NODE_IRREGULAR=$GPUS_PER_NODE_IRREGULAR
    export GPUS_PER_NODE="" # set to empty string
fi
# either GPUS_PER_NODE_IRREGULAR or GPUS_PER_NODE must be set.
if [ -z "$GPUS_PER_NODE_IRREGULAR" ] && [ -z "$GPUS_PER_NODE" ]; then
    echo "Error: Either GPUS_PER_NODE_IRREGULAR or GPUS_PER_NODE must be set"
    exit 1
fi
# cannot be set both
if [ ! -z "$GPUS_PER_NODE_IRREGULAR" ] && [ ! -z "$GPUS_PER_NODE" ]; then
    echo "Error: Cannot be set both GPUS_PER_NODE_IRREGULAR=$GPUS_PER_NODE_IRREGULAR and GPUS_PER_NODE=$GPUS_PER_NODE"
    exit 1
fi
# end


module load rocm/6.2.1

bash make_hostfile_irregular.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-13701}

cd Megatron-DeepSpeed
bash pretrain_gpt_on_amd_ds_irregular_devicemesh.sh
