TASK_NAME=med_full
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed --num_gpus=4 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /mandapeng16/lq/med_data/train_med.json \
    --test_file /mandapeng16/lq/med_data/dev_med.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /mandapeng16/lq/chatglm-6b \
    --output_dir /mandapeng16/lq/output_med_full/adgen-chatglm-6b-ft-$LR \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --task_name $TASK_NAME \
    --base_cache_dir /mandapeng16/lq/cache \
    --fp16

