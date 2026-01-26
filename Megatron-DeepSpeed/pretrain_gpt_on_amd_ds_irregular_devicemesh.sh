#!/bin/bash

# Start timer
START_TIME=$(date +%s)

# Multi-node training configuration
NNODES=${SLURM_NNODES:-1}                     # Number of nodes

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

## Replace with your own dataset path
BASE_DATA_PATH="path/to/dataset"
DATASET=${BASE_DATA_PATH}/text_document
VOCAB_PATH=${BASE_DATA_PATH}/vocab.json
MERGE_PATH=${BASE_DATA_PATH}/merges.txt
##

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_configs/ds_config_${SLURM_JOB_ID}.json"
HF="$DIR/hostfiles/hostfile_${SLURM_JOB_ID}"

DTYPE=${DTYPE:-"bf16"}

SIZE_TAG=${SIZE_TAG:-"1B"}
EXIT_INTERVAL=${EXIT_INTERVAL:-100}

# Profiler configuration (passed from launcher via environment variables)
# Profiler env vars (schedule params are hardcoded in training.py)
export PROFILER_ENABLED="${PROFILER_ENABLED:-0}"
export PROFILER_TRACE_DIR="${PROFILER_TRACE_DIR:-profiler_traces}"
export PROFILER_TRACE_PREFIX="${PROFILER_TRACE_PREFIX:-trace}"

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

# MICRO_BATCH=ceil(GLOBAL_BATCH / (DP * GRAD_ACCUM_STEPS))
GLOBAL_BATCH=${GLOBAL_BATCH:-8}
DIVISOR=$((DP * GRAD_ACCUM_STEPS))
MICRO_BATCH=${MICRO_BATCH:-$(((GLOBAL_BATCH + DIVISOR - 1) / DIVISOR))}

TRAIN_ITERS=${TRAIN_ITERS:-100000}
LR=${LR:-6.0e-3}
MIN_LR=${MIN_LR:-6.0e-4}

# Add memory logging options
MEMORY_LOGGING=${MEMORY_LOGGING:-1}  # Enable memory logging by default

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
      --${DTYPE} \
      --no-gradient-accumulation-fusion \
      --no-masked-softmax-fusion \
      --use-flash-attn-v2 \
    "
    ##############################################################
    # previous `pretrain_gpt_on_amd_ds_irregular_devicemesh_debug.sh` has these options
      # --timing-log-level 2 \
      # --no-pipeline-parallel \
    # --tensorboard-dir ${TENSORBOARD_DIR} \
    ##############################################################
    # --checkpoint-activations \
      # --no-bias-gelu-fusion \
      # --no-bias-dropout-fusion \

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
        # --deepspeed-activation-checkpointing \
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

# Add profiler options if enabled
if [ "$PROFILER_ENABLED" == "1" ]; then
    options="${options} --profiler-enabled --profiler-trace-dir $PROFILER_TRACE_DIR --profiler-trace-prefix $PROFILER_TRACE_PREFIX"
fi
    
# Create DeepSpeed configuration with multi-node settings
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
  
  "data_types": {
    "grad_accum_dtype": "fp32" 
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
# , Total GPUs: $TOTAL_GPUS"
echo "============= MULTI-NODE TRAINING CONFIGURATION ============="
echo "Nodes: $NNODES, GPUs per node: $GPUS_PER_NODE, GPUS_PER_NODE_IRREGULAR: $GPUS_PER_NODE_IRREGULAR"
echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Model size: $SIZE_TAG (Layers=$LAYERS, Hidden=$HIDDEN)"
echo "Parallelism: TP=$TP, PP=$PP, DP=$DP, SP=$SP, ZERO=$ZERO_STAGE, ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING"
echo "Batch sizes: Global=$GLOBAL_BATCH, Micro=$MICRO_BATCH, Grad Accum Steps=$GRAD_ACCUM_STEPS, Data type: $DTYPE"
echo "Profiler: PROFILER_ENABLED=$PROFILER_ENABLED, PROFILER_TRACE_DIR=$PROFILER_TRACE_DIR"
echo "==========================================================="

#  WORKER_STR="--num_nodes $NUM_NODES --num_gpus $WORLD_SIZE"

run_cmd="deepspeed --hostfile $HF --master_addr $MASTER_ADDR --master_port $MASTER_PORT $WORKER_STR ${DIR}/pretrain_gpt.py $@ ${options}"

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

set +x
