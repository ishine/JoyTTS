#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

#stage=1只训练llm-tts，2训练llm-chat+llm-tts
stage=1

pretrained_model_dir=../../pretrained_models


# train llm
export CUDA_VISIBLE_DEVICES="6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
# train_engine=deepspeed
finetune_model="llm" #"llm flow"
if [ $stage == 1 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi

  cat data/parquet/RedGPT-main/train/data.list \
      data/parquet/generated_chat_0.4M/train/data.list \
      data/parquet/RedGPT-main_augment/train/data.list \
      data/parquet/generated_chat_0.4M_augment/train/data.list \
      > data/parquet/train_data.list
  cat data/parquet/RedGPT-main/eval/data.list \
      > data/parquet/eval_data.list
  # NOTE will update llm/hift training later
  for model in $finetune_model; do
    # torchrun --nnodes=1 --nproc_per_node=1 \
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_llm.yaml \
      --train_data data/parquet/train_data.list \
      --cv_data data/parquet/eval_data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --chat_pretrain_path $pretrained_model_dir/Chat \
      --model $model \
      --checkpoint $pretrained_model_dir/$model-checkpoint.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model_only
  done
fi


# train end2ene
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
# train_engine=torch_ddp
train_engine=deepspeed
finetune_model="llm" #"llm flow"
if [ $stage == 2 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi

  cat data/parquet_uselabel/RedGPT-main/train/data.list \
      > data/parquet/train_data.list
  cat data/parquet_uselabel/RedGPT-main/eval/data.list \
      > data/parquet/eval_data.list
  # NOTE will update llm/hift training later
  for model in $finetune_model; do
    # torchrun --nnodes=1 --nproc_per_node=1 \
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_end2end.yaml \
      --train_data data/parquet/train_data.list \
      --cv_data data/parquet/eval_data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --chat_pretrain_path $pretrained_model_dir/Chat \
      --model $model \
      --checkpoint $pretrained_model_dir/$model-checkpoint.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model_only
  done
fi

# average model
average_num=5
if [ $stage == 3 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

#Export model
if [ $stage == 4 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi

