import gradio as gr
import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_PATH = "/mandapeng16/lq/chatglm-6b"
CHECKPOINT_PATH = "/mandapeng16/lq/output_med2/adgen-chatglm-6b-pt-128-1e-2/checkpoint-3000"

# Load the tokenizer for ChatGLM
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Load the fine-tuned model
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, pre_seq_len=128)
model_finetuned = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True).cuda()

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}

for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model_finetuned.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model_finetuned = model_finetuned.quantize(8)
model_finetuned = model_finetuned.half().cuda()
model_finetuned.transformer.prefix_encoder.float()
model_finetuned = model_finetuned.eval()


# Load the original model
model_original = AutoModel.from_pretrained("/mandapeng16/lq/chatglm-6b", trust_remote_code=True).half().cuda().eval()

# Function for model inference
def chatglm_outputs(input_text):
    response_original = model_original.chat(tokenizer, input_text, history=[])
    response_finetuned = model_finetuned.chat(tokenizer, input_text, history=[])
    return response_original[0], response_finetuned[0]

# Gradio interface
# textbox = gr.inputs.Textbox(lines=5, label="Input Text")

iface = gr.Interface(fn=chatglm_outputs,
                     inputs=gr.Textbox(lines=5, placeholder="请输入问题", label="用户"),
                     # outputs=["text", "text"],
                     outputs=[
                         gr.Textbox(label="ChatGLM-6B", type="text"),
                         gr.Textbox(label="ChatGLM-6B-med", type="text")
                     ],
                     title="ChatGLM-6B vs ChatGLM-6B-med",
                     description="对比微调前后的模型问答"
                    )

iface.launch()
