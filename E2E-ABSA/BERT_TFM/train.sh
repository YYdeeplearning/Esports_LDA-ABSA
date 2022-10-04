#!/usr/bin/env bash
TASK_NAME=esports
ABSA_TYPE=crf

CUDA_VISIBLE_DEVICES=0,2,3 python main.py --model_type bert \
                         --absa_type tfm \
                         --tfm_mode finetune \
                         --fix_tfm 0 \
                         --model_name_or_path bert-base-uncased \
                         --data_dir ./data/esports \
                         --task_name esports \
                         --per_gpu_train_batch_size 16 \
                         --per_gpu_eval_batch_size 8 \
                         --learning_rate 2e-5 \
                         --do_train \
                         --do_eval \
                         --do_lower_case \
                         --tagging_schema BIEOS \
                         --overfit 0 \
                         --overwrite_output_dir \
                         --eval_all_checkpoints \
                         --MASTER_ADDR localhost \
                         --MASTER_PORT 28512 \
                         --max_steps 1500
