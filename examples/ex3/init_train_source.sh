#!/bin/bash


DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
BASE_DATA_PATH=$UCP_DIR/training_datasets
echo "BASE_DATA_PATH: $BASE_DATA_PATH"
DATASET=${BASE_DATA_PATH}/gpt2_text_document
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
echo "VOCAB_PATH: $VOCAB_PATH"
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt


script_path=$(realpath $0)
script_dir=$(dirname $script_path)
CONFIG_JSON="$script_dir/ds_config.json"

ZERO_STAGE=1
DTYPE="fp16"

# Model size
LAYERS=12
HIDDEN=768
SEQ=1024
ATTN_HEADS=12
EXIT_INTERVAL=200
SIZE_TAG="125M"

# 3D parallelism of training 
TP=1
PP=1
DP=2
SP=1
WORLD_SIZE=$((TP*PP*DP*SP))
GLOBAL_BATCH=4
MICRO_BATCH=$((GLOBAL_BATCH/WORLD_SIZE))
TRAIN_ITERS=100000
LR=2.0e-3
MIN_LR=2.0e-4

# 3D parallelism of checkpoint to load
LOAD_TP=$TP
LOAD_PP=$PP
LOAD_DP=$DP
LOAD_SP=$SP
RUN_TAG="save"
# RUN_TAG="ref_load${LOAD_TP}_${LOAD_PP}_${LOAD_DP}"

CHECKPOINT_PATH=checkpoints/gpt2/z${ZERO_STAGE}/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_${SIZE_TAG}
LOAD_CHECKPOINT_PATH=checkpoints/gpt2/z${ZERO_STAGE}/$DTYPE/tp${LOAD_TP}_pp${LOAD_PP}_dp${LOAD_DP}_sp${LOAD_SP}_${SIZE_TAG}
LOG_DIR="tensorboard/$DTYPE/tp${TP}_pp${PP}_dp${DP}_sp${SP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_${SIZE_TAG}_${RUN_TAG}"
mkdir -p $LOG_DIR

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -z|--zero-stage)
    ZERO_STAGE=$2;
    shift
    ;;
    *)
    echo "Unknown argument(s)"
    usage
    exit 1
    shift
    ;;
esac
done


options=" \
	--tensor-model-parallel-size $TP \
	--pipeline-model-parallel-size $PP \
    --ds-sequence-parallel-size $SP \
        --num-layers $LAYERS \
        --hidden-size $HIDDEN \
        --num-attention-heads $ATTN_HEADS \
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
        --eval-iters 40 \
        --eval-interval 10 \
	--data-path ${DATASET} \
	--vocab-file ${VOCAB_PATH} \
	--merge-file ${MERGE_PATH} \
	--save-interval 100 \
        --split 98,2,0 \
        --clip-grad 1.0 \
	--weight-decay 0.1 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--init-method-std 0.006 \
        --${DTYPE} \
	--checkpoint-activations \
	--exit-interval ${EXIT_INTERVAL} \
        --save ${CHECKPOINT_PATH} \
        --load ${LOAD_CHECKPOINT_PATH} \
        --make-vocab-size-divisible-by 256 \
	--tensorboard-dir $LOG_DIR
        "

options="${options} \
        --deepspeed \
        --deepspeed_config=${CONFIG_JSON} \
        --zero-stage=${ZERO_STAGE} \
        --deepspeed-activation-checkpointing \
"
if [[ ${ZERO_STAGE} -gt 1 ]]; then
options="${options} \
    --no-pipeline-parallel"
fi

cat <<EOT > $CONFIG_JSON
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": false
  },

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 50,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : false
}
EOT

WORKER_STR="--num_nodes 1 --num_gpus $WORLD_SIZE"
run_cmd="deepspeed --master_port 29700 $WORKER_STR ${MEGATRON_DEEPSPEED_DIR}/pretrain_gpt.py $@ ${options}"


echo ${options}
echo ${run_cmd}
eval ${run_cmd}

set +x