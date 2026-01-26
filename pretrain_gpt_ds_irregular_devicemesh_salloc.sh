#!/bin/bash

set -e

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Salloc mode training started at: $DATETIME"

# Validate CLUSTER
if [ -z "$CLUSTER" ]; then
    echo "Error: CLUSTER environment variable is not set. Must be 'delta' or 'amd'"
    exit 1
fi

if [ "$CLUSTER" != "delta" ] && [ "$CLUSTER" != "amd" ]; then
    echo "Error: CLUSTER must be 'delta' or 'amd', got: $CLUSTER"
    exit 1
fi

echo "Running on cluster: $CLUSTER"

# Validate required environment variables
required_vars=("HOSTFILE_PATH" "MASTER_ADDR" "SLURM_NNODES" "TP" "PP" "DP" "SP" "GLOBAL_BATCH" "ZERO_STAGE" "SIZE_TAG")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# Validate hostfile exists
if [ ! -f "$HOSTFILE_PATH" ]; then
    echo "Error: Hostfile not found at $HOSTFILE_PATH"
    exit 1
fi

echo "Using hostfile: $HOSTFILE_PATH"
cat "$HOSTFILE_PATH"

export EXIT_INTERVAL=${EXIT_INTERVAL:-5}

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

export OMP_NUM_THREADS=1

# Set default MASTER_PORT if not provided
export MASTER_PORT=${MASTER_PORT:-13701}

# Copy hostfile to expected location for the training script
# The training script expects hostfile at Megatron-DeepSpeed/hostfiles/hostfile_${SLURM_JOB_ID}
# For salloc mode, we use a pseudo job ID based on timestamp
export SLURM_JOB_ID="salloc_$(date +%s)"
mkdir -p Megatron-DeepSpeed/hostfiles
cp "$HOSTFILE_PATH" "Megatron-DeepSpeed/hostfiles/hostfile_${SLURM_JOB_ID}"

# Set up logging if LOG_DIR is specified
if [ ! -z "$LOG_DIR" ]; then
    export LOG_TO_FILE=1
    export LOG_DIR="$LOG_DIR"
    export LOG_FILENAME="${LOG_FILENAME:-log.txt}"
    mkdir -p "$LOG_DIR"
    echo "Logs will be saved to: $LOG_DIR/$LOG_FILENAME"
fi

# Export profiler environment variables (passed from Python launcher)
# Profiler env vars (schedule params are hardcoded in training.py)
export PROFILER_ENABLED="${PROFILER_ENABLED:-0}"
export PROFILER_TRACE_DIR="${PROFILER_TRACE_DIR:-}"
export PROFILER_TRACE_PREFIX="${PROFILER_TRACE_PREFIX:-trace}"
echo "Profiler config: PROFILER_ENABLED=$PROFILER_ENABLED, PROFILER_TRACE_DIR=$PROFILER_TRACE_DIR"

echo "============================================"
echo "Salloc Training Configuration:"
echo "  CLUSTER: $CLUSTER"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  SLURM_NNODES: $SLURM_NNODES"
echo "  SLURM_JOB_ID (pseudo): $SLURM_JOB_ID"
echo "  LOG_DIR: ${LOG_DIR:-<not set, logs to stdout>}"
echo "============================================"

cd Megatron-DeepSpeed

# Select training script based on cluster
if [ "$CLUSTER" == "delta" ]; then
    echo "Running Delta (NVIDIA) training script..."
    bash pretrain_gpt_on_delta_ds_irregular_devicemesh.sh
elif [ "$CLUSTER" == "amd" ]; then
    echo "Running AMD training script..."
    bash pretrain_gpt_on_amd_ds_irregular_devicemesh.sh
fi
