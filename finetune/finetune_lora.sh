#!/bin/bash

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
 
MODEL="../pretrained_models/Chat"

DATANAME='RedGPT-main'
DATA="../../data/llm/$DATANAME/train.json"
EVAL_DATA="../../data/llm/$DATANAME/eval.json"

LLM_TYPE="qwen"   
MODEL_MAX_Length=2048

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)" \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --eval_steps 1000 \
    --output_dir output/output_lora \
    --logging_dir output/output_lora \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "tensorboard"
