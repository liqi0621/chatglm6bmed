PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm-6b-pt-128-1e-2
STEP=3000
TASK_NAME=med2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_predict \
    --validation_file /mandapeng16/lq/med_data/dev_med.json \
    --test_file /mandapeng16/lq/med_data/dev_med.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /mandapeng16/lq/chatglm-6b \
    --ptuning_checkpoint /mandapeng16/lq/output_med2/adgen-chatglm-6b-pt-128-1e-2/checkpoint-3000 \
    --output_dir /mandapeng16/lq/output_med2/adgen-chatglm-6b-pt-128-1e-2/checkpoint-3000 \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8 \
    --task_name $TASK_NAME \
    --base_cache_dir /mandapeng16/lq/cache
