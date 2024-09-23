TASK_NAME=med_lora
# PRE_SEQ_LEN=128
PEFT_TYPE=lora
LORA_DIM=8
LR=1e-2

CHAT_TRAIN_DATA=/mandapeng16/lq/med_data/train_med.json
CHAT_VAL_DATA=/mandapeng16/lq/med_data/dev_med.json

MODEL_NAME_OR_PATH=/mandapeng16/lq/chatglm-6b

NUM_GPUS=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CUDA_VISIBLE_DEVICES=0
#deepspeed --num_gpus=$NUM_GPUS --master_port $MASTER_PORT chatglm_model_v1/run_peft.py \
CUDA_VISIBLE_DEVICES=0 python3 run_peft.py \
    --do_train \
    --train_file $CHAT_TRAIN_DATA \
    --test_file $CHAT_VAL_DATA \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir /mandapeng16/lq/output_medlora/chatglm-6b-$TASK_NAME-$PEFT_TYPE-$LORA_DIM-$LR \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --peft_type $PEFT_TYPE \
    --lora_dim $LORA_DIM \
    --task_name $TASK_NAME \
    --base_cache_dir /mandapeng16/lq/cache \
    --quantization_bit 8 \
    --fp16
    # --overwrite_cache \
