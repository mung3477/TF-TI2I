import os
os.environ["HF_HOME"] = "/root/Desktop/workspace/woosung/.hub"

import PIL
from PIL import Image
from diffusers.utils import make_image_grid
from diffusers import StableDiffusion3Pipeline
from src.customized_pipe import TI2I_StableDiffusion3Pipeline
from src.attn_processor import TI2I_JointAttnProcessor2_0_multi
import numpy as np
import gradio as gr
import torch

def load_model():
    global pipe
    pipe = TI2I_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                                torch_dtype=torch.float16,
                                                                device_map="balanced")
    return "Model loaded successfully!"

def ti2i_gen(prompt, img1, use_img1,sub_p1, img2, use_img2,sub_p2, img3, use_img3,sub_p3, img4, use_img4,sub_p4, c_layer, seed, wta_control,rcm_control):

    def to_pil(image):
        if isinstance(image, np.ndarray):
            return PIL.Image.fromarray(image)
        return image  # Already a PIL image
    refer_images = []
    refer_prompts=[]
    if use_img1 and img1 is not None:
        refer_images.append(to_pil(img1).resize((1024,1024)))
        refer_prompts.append(sub_p1)
    if use_img2 and img2 is not None:
        refer_images.append(to_pil(img2).resize((1024,1024)))
        refer_prompts.append(sub_p2)
    if use_img3 and img3 is not None:
        refer_images.append(to_pil(img3).resize((1024,1024)))
        refer_prompts.append(sub_p3)
    if use_img4 and img4 is not None:
        refer_images.append(to_pil(img4).resize((1024,1024)))
        refer_prompts.append(sub_p4)

    num_refer=len(refer_images)
    layer_count = 0
    attn_processors = []
    def iter_net(net):
        nonlocal layer_count
        for child in net.children():
            if "Attention" in child.__class__.__name__:
                child.processor = TI2I_JointAttnProcessor2_0_multi(layer=layer_count, contextual_replace=True,
                    wta_control_signal={"on":wta_control,
                                        "hyper_parameter":{
                                                        "wta_weight":[1]*num_refer,
                                                        "cross2ref":True,
                                                        "wta_shift":[0]*num_refer,
                                                        "wta_cross":False
                                                        },
                                        "debug":False},
                    ref_control_signal={"on":rcm_control,
                                        "ref_idxs":[i for i in range(num_refer)],
                                        "control_type":"main_context",
                                        "debug":False,
                                        "control_layers":[i for i in range(c_layer,40)],
                                        "hyper_parameter":{}},
                                        )
                attn_processors.append(child.processor)
                layer_count += 1
            iter_net(child)

    iter_net(pipe.transformer)
    torch.manual_seed(seed)
    pipe.enable_attention_slicing()
    switch_images = pipe.img2img_multi(
    images=refer_images,
    prompt_a=prompt,
    prompt_b=refer_prompts,
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=5,
    strength=1,
    height=1024,
    width=1024,
    )

    return switch_images[0]

with gr.Blocks() as demo:
    gr.Markdown("# TF-TI2I 📃➕🖼️➡️🖼️")

    with gr.Row():
        with gr.Column(scale=1):
            load_button = gr.Button("Load Model")
            load_status = gr.Textbox(label="Model Status", interactive=False)
            prompt = gr.Textbox(label="Enter Prompt📃",value="a dinosaur with the texture of crystal doing breathing fire in Starry Night")
            clayer_slider = gr.Slider(0, 40, label="Layer of start TF-TI2I control", value=25)
            seed_slider = gr.Slider(0, 100000, label="Random seed", value=0)
            with gr.Row():
                wta_control = gr.Checkbox(label="Enable Winner Takes All (Recommend)",value=True)
                rcm_control = gr.Checkbox(label="Enable Reference Contextual Masking (Disable for efficiency with minor loss)",value=True)

        with gr.Column(scale=1):
            output = gr.Image(label="Generated Image")
            submit_button = gr.Button("Generate Image🔥")

    with gr.Row():
        with gr.Column(scale=1):
            use_img1 = gr.Checkbox(label="Use this image as reference🖼️",value=True)
            sub_p1 = gr.Textbox(label="Sub Prompt for reference📃(optional) ",value="a dinosaur")
            img1 = gr.Image(label="Image 1", value=Image.open("refer_data/A dinosaur.png").resize((1024,1024)))

        with gr.Column(scale=1):
            use_img2 = gr.Checkbox(label="Use this image as reference🖼️",value=True)
            sub_p2 = gr.Textbox(label="Sub Prompt for reference📃(optional) ",value="crystal")
            img2 = gr.Image(label="Image 2", value=Image.open("refer_data/crystal dog.png").resize((1024,1024)))

        with gr.Column(scale=1):
            use_img3 = gr.Checkbox(label="Use this image as reference🖼️",value=True)
            sub_p3 = gr.Textbox(label="Sub Prompt for reference📃(optional) ",value="breathing fire")
            img3 = gr.Image(label="Image 3",value=Image.open("refer_data/breathing fire.png").resize((1024,1024)))

        with gr.Column(scale=1):
            use_img4 = gr.Checkbox(label="Use this image as reference🖼️",value=True)
            sub_p4 = gr.Textbox(label="Sub Prompt for reference📃(optional) ",value="Starry Night")
            img4 = gr.Image(label="Image 4",value=Image.open("refer_data/ood_starry.jpg").resize((1024,1024)))




    load_button.click(load_model, outputs=load_status)
    submit_button.click(ti2i_gen,
                        inputs=[prompt, img1, use_img1,sub_p1, img2, use_img2,sub_p2, img3, use_img3,sub_p3, img4, use_img4,sub_p4, clayer_slider, seed_slider, wta_control,rcm_control],
                        outputs=output)

demo.launch()
