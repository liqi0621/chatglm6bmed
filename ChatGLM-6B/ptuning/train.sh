TASK_NAME=med2
PRE_SEQ_LEN=128
LR=1e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file /mandapeng16/lq/med_data/train_med.json \
    --validation_file /mandapeng16/lq/med_data/dev_med.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /mandapeng16/lq/chatglm-6b \
    --output_dir /mandapeng16/lq/output_med2/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
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
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8 \
    --task_name $TASK_NAME \
    --base_cache_dir /mandapeng16/lq/cache
