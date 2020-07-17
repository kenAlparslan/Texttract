#!/usr/bin/bash

python run_classification.py    --data_dir data \
                                --model_type layoutlm \
                                --model_name_or_path ./layoutlm-base-uncased \
                                --do_lower_case \
                                --max_seq_length 512 \
                                --do_train false \
                                --do_eval false \
                                --do_test true\
                                --num_train_epochs 100.0 \
                                --logging_steps 10 \
                                --save_steps -1 \
                                --output_dir ./output/ \
                                --labels data/labels.txt \ 
                                --per_gpu_train_batch_size 16 \
                                --per_gpu_eval_batch_size 16 \
                                --fp16