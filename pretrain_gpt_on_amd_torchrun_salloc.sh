#!/bin/bash

set -e

# Initialize conda and activate environment
source /path/to/conda/env/etc/profile.d/conda.sh
conda activate ucx
## Replace with your own dataset path
BASE_DATA_PATH="path/to/dataset"
DATASET=${BASE_DATA_PATH}/text_document
VOCAB_PATH=${BASE_DATA_PATH}/vocab.json
MERGE_PATH=${BASE_DATA_PATH}/merges.txt
##


# Start timer
START_TIME=$(date +%s)

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Single-node torchrun training started at: $DATETIME on node $(hostname)"

# Single node configuration
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_ADDR=localhost
MASTER_PORT=${MASTER_PORT:-13701}

# Validate required environment variables
required_vars=("GPUS_PER_NODE" "TP" "PP" "DP" "SP" "GLOBAL_BATCH" "ZERO_STAGE" "SIZE_TAG")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: Required environment variable $var is not set"
        exit 1
    fi
done

# Generate pseudo SLURM_JOB_ID for salloc mode
export SLURM_JOB_ID=${SLURM_JOB_ID:-"salloc_$(date +%s)"}

export OMP_NUM_THREADS=1

# Change to Megatron-DeepSpeed directory
cd Megatron-DeepSpeed

DIR=`pwd`

CONFIG_JSON="$DIR/ds_configs/ds_config_${SLURM_JOB_ID}.json"

DTYPE=${DTYPE:-"bf16"}
EXIT_INTERVAL=${EXIT_INTERVAL:-100}

if [[ $SIZE_TAG == "1B" ]]; then
    LAYERS=24
    HIDDEN=2048
    SEQ=2048
    NUM_ATTN_HEADS=32
elif [[ $SIZE_TAG == "3B" ]]; then
    LAYERS=32
    HIDDEN=2560
    SEQ=2048
    NUM_ATTN_HEADS=32
elif [[ $SIZE_TAG == "7B" ]]; then
    LAYERS=32
    HIDDEN=4096
    SEQ=2048
    NUM_ATTN_HEADS=32
elif [[ $SIZE_TAG == "13B" ]]; then
    LAYERS=40
    HIDDEN=5120
    SEQ=2048
    NUM_ATTN_HEADS=40
elif [[ $SIZE_TAG == "30B" ]]; then
    LAYERS=64
    HIDDEN=6144
    SEQ=2048
    NUM_ATTN_HEADS=48
elif [[ $SIZE_TAG == "40B" ]]; then
    LAYERS=64
    HIDDEN=8192
    SEQ=2048
    NUM_ATTN_HEADS=64
elif [[ $SIZE_TAG == "65B" ]]; then
    LAYERS=80
    HIDDEN=8192
    SEQ=2048
    NUM_ATTN_HEADS=64
fi

# 3D parallelism of training
TP=${TP:-1}
PP=${PP:-1}
DP=${DP:-1}
SP=${SP:-1}
ZERO_STAGE=${ZERO_STAGE:-0}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-1}

# Calculate total world size
TOTAL_GPUS=$GPUS_PER_NODE
WORLD_SIZE=$((TP*PP*DP*SP))

# Assert that requested parallelism exactly matches available GPUs
if [ $WORLD_SIZE -ne $TOTAL_GPUS ]; then
    echo "Error: Parallelism configuration (TP=$TP, PP=$PP, DP=$DP, SP=$SP) must use exactly $TOTAL_GPUS GPUs."
    exit 1
fi

# MICRO_BATCH=ceil(GLOBAL_BATCH / (DP * GRAD_ACCUM_STEPS))
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
DIVISOR=$((DP * GRAD_ACCUM_STEPS))
MICRO_BATCH=${MICRO_BATCH:-$(((GLOBAL_BATCH + DIVISOR - 1) / DIVISOR))}

TRAIN_ITERS=${TRAIN_ITERS:-100000}
LR=${LR:-6.0e-3}
MIN_LR=${MIN_LR:-6.0e-4}

# Add memory logging options
MEMORY_LOGGING=${MEMORY_LOGGING:-1}

# Create directories for logs and tensorboard
RUN_TAG="z${ZERO_STAGE}_${DTYPE}_tp${TP}_pp${PP}_dp${DP}_sp${SP}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_nodes${NNODES}"
TENSORBOARD_DIR="tensorboard/${SIZE_TAG}/${RUN_TAG}/${DATETIME}"
mkdir -p $TENSORBOARD_DIR

options=" \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --ds-sequence-parallel-size $SP \
    --data-parallel-size-arg $DP \
    --num-layers $LAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $NUM_ATTN_HEADS \
    --seq-length $SEQ \
    --loss-scale 12 \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --gradient-accumulation-steps $GRAD_ACCUM_STEPS \
    --train-iters $TRAIN_ITERS \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 1000 \
    --eval-interval 1000 \
    --data-path ${DATASET} \
    --vocab-file ${VOCAB_PATH} \
    --merge-file ${MERGE_PATH} \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.006 \
    --exit-interval ${EXIT_INTERVAL} \
    --make-vocab-size-divisible-by 256 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --${DTYPE} \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --use-flash-attn-v2 \
    "

# Add memory logging options if enabled
if [ $MEMORY_LOGGING -eq 1 ]; then
    options="${options} \
    --log-memory-to-tensorboard"
fi

options="${options} \
    --deepspeed \
    --deepspeed_config=${CONFIG_JSON} \
    --zero-stage=${ZERO_STAGE} \
"

if [ ${ZERO_STAGE} -gt 1 ]; then
    options="${options} \
    --no-pipeline-parallel"
fi

if [ ${ACTIVATION_CHECKPOINTING} -eq 1 ]; then
    options="${options} \
    --checkpoint-activations \
    --deepspeed-activation-checkpointing \
"
fi

# Create DeepSpeed configuration
mkdir -p $DIR/ds_configs
cat <<EOT > $CONFIG_JSON
{
  "train_batch_size": $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,
  "gradient_accumulation_steps": $GRAD_ACCUM_STEPS,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "overlap_comm": true,
    "use_multi_rank_bucket_allreduce": false,
    "reduce_bucket_size": 50000000
  },

  "bf16": {
    "enabled": true
  },

  "wall_clock_breakdown": false,

  "monitor_config": {
    "enabled": true,
    "monitor_memory": true,
    "monitor_gpu_utilization": true
  },

  "memory_breakdown": false
}
EOT

# Print configuration summary
echo "============= SINGLE-NODE TORCHRUN CONFIGURATION (AMD) ============="
echo "Node: $(hostname)"
echo "GPUs: $GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Model size: $SIZE_TAG (Layers=$LAYERS, Hidden=$HIDDEN)"
echo "Parallelism: TP=$TP, PP=$PP, DP=$DP, SP=$SP, ZERO=$ZERO_STAGE, ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING"
echo "Batch sizes: Global=$GLOBAL_BATCH, Micro=$MICRO_BATCH, Grad Accum Steps=$GRAD_ACCUM_STEPS, Data type: $DTYPE"
echo "===================================================================="

LAUNCHER="torchrun \
  --standalone \
  --nproc_per_node=${GPUS_PER_NODE}"

# Create the training command
run_cmd="${LAUNCHER} ${DIR}/pretrain_gpt.py ${options} $@ 2>&1"

# Log to file if requested
LOG_TO_FILE=${LOG_TO_FILE:-0}
LOG_DIR=${LOG_DIR:-"logs/${SIZE_TAG}/${RUN_TAG}/${DATETIME}"}
LOG_FILENAME=${LOG_FILENAME:-"log.txt"}
if [ ${LOG_TO_FILE} -eq 1 ]; then
    mkdir -p $LOG_DIR
    run_cmd="${run_cmd} | tee ${LOG_DIR}/${LOG_FILENAME}"
fi

# Run the training command
echo "Running command:"
echo ${run_cmd}

eval ${run_cmd}

# End timer and show duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "Training completed in ${DURATION}s"
