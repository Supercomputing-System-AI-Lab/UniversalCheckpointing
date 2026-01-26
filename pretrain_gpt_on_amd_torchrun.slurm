#!/bin/bash
#SBATCH -J pretrain-gpt-debug          # Job name
#SBATCH -o slurm_logs/%x_%j.out      # Name of stdout output file (%j expands to jobId)
#SBATCH -N 2                     # Total number of nodes requested
#SBATCH -t 00:04:00              # Run time (hh:mm:ss)
#SBATCH -p mi2508x               # Partition

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo $DATETIME

# Run SLURM commands on host to get cluster information
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-13701}

export EXIT_INTERVAL=5

# Calculate total parallelism
TOTAL_PARALLELISM=$((TP * PP * DP * SP))

# Check if total parallelism is divisible by node count
if (( TOTAL_PARALLELISM % SLURM_NNODES != 0 )); then
    echo "Error: Total parallelism (TP*PP*DP*SP = $TOTAL_PARALLELISM) must be divisible by node count ($SLURM_NNODES)"
    exit 1
fi

export GPUS_PER_NODE=$((TOTAL_PARALLELISM / SLURM_NNODES))

# Validate GPUS_PER_NODE
if [ "$GPUS_PER_NODE" -le 0 ]; then
    echo "Error: GPUS_PER_NODE must be greater than 0: $GPUS_PER_NODE"
    exit 1
fi

# Set CUDA_VISIBLE_DEVICES if not using all GPUs (for NVIDIA/CUDA)
if [ "$GPUS_PER_NODE" -lt 8 ]; then
    # Create a comma-separated list of device IDs from 0 to GPUS_PER_NODE-1
    DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))
    export CUDA_VISIBLE_DEVICES=$DEVICES
    echo "Setting CUDA_VISIBLE_DEVICES=$DEVICES"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1
module load rocm/6.2.1
export OMP_NUM_THREADS=1

# Execute the training workload within the Apptainer container
# Use srun to launch the containerized training across nodes
# Explicitly pass critical environment variables to ensure they reach the container
srun --export=ALL bash -c '
cd Megatron-DeepSpeed
export LOG_TO_FILE=0
bash pretrain_gpt_on_amd_torchrun.sh
'
