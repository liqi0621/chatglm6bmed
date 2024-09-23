import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_PATH = "/mandapeng16/lq/chatglm-6b"
CHECKPOINT_PATH = "/mandapeng16/lq/output_Ped1st/adgen-chatglm-6b-pt-128-1e-6/checkpoint-3000"

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True).cuda()

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}

for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)



print(f"Quantized to 8 bit")
model = model.quantize(8)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()


print("用户：你好\n")
response, history = model.chat(tokenizer, "你好", history=[])
print("ChatGLM-6B：\n",response)
print("\n------------------------------------------------\n用户：")

line = input()
while line:
    response, history = model.chat(tokenizer, line, history=history)
    print("ChatGLM-6B：\n", response)
    print("\n------------------------------------------------\n用户：")
    line = input()
