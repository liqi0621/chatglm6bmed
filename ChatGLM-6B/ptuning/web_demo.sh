PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python3 web_demo.py \
    --model_name_or_path /mandapeng16/lq/chatglm-6b \
    --ptuning_checkpoint /mandapeng16/lq/output_med2/adgen-chatglm-6b-pt-128-1e-2/checkpoint-3000 \
    --pre_seq_len $PRE_SEQ_LEN

