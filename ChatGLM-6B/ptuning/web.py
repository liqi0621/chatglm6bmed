import gradio as gr
import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL_PATH = "/mandapeng16/lq/chatglm-6b"
CHECKPOINT_PATH = "/mandapeng16/lq/output_Ped1st/adgen-chatglm-6b-pt-128-1e-6/checkpoint-3000"

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


model1 = AutoModel.from_pretrained("/mandapeng16/lq/chatglm-6b", trust_remote_code=True).half().cuda()
model1 = model1.eval()


config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, pre_seq_len=128)
model2 = AutoModel.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True).cuda()

prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}

for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model2.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# print(f"Quantized to 8 bit")
model2 = model2.quantize(8)
model2 = model2.half().cuda()
model2.transformer.prefix_encoder.float()
model2 = model2.eval()


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict1(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []
    for response, history in model1.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="用户：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates


def predict2(input, max_length, top_p, temperature, history=None):
    if history is None:
        history = []
    for response, history in model2.stream_chat(tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        updates = []
        for query, response in history:
            updates.append(gr.update(visible=True, value="用户：" + query))
            updates.append(gr.update(visible=True, value="ChatGLM-6B-med：" + response))
        if len(updates) < MAX_BOXES:
            updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
        yield [history] + updates


with gr.Blocks() as demo:
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="提问："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="回复："))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", lines=11).style(
                container=False)
        with gr.Column(scale=1):
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            button = gr.Button("Generate")
    button.click(predict1, [txt, max_length, top_p, temperature, state], [state] + text_boxes, predict2, [txt, max_length, top_p, temperature, state], [state] + text_boxes)

demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=7860)




# greeter_1 = gr.Interface(fn=predict1, inputs="textbox", outputs=gr.Textbox(label="ChatGLM-6B"))
# greeter_2 = gr.Interface(fn=predict2, inputs="textbox", outputs=gr.Textbox(label="ChatGLM-6B-med"))

# demo = gr.Parallel(greeter_1, greeter_2)

#if __name__ == "__main__":
#    demo.launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=7860)

