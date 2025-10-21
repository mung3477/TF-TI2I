import os
os.environ["HF_HOME"] = "/root/Desktop/workspace/yujin/woosung/.hub"

import numpy as np
import torch

import re
from tqdm import tqdm
import PIL
from PIL import Image
from diffusers.utils import make_image_grid
import sys

import dotenv
from huggingface_hub import login

dotenv.load_dotenv()
login(os.getenv("HF_TOKEN"))

from src.customized_pipe import TI2I_StableDiffusion3Pipeline
from src.attn_processor import TI2I_JointAttnProcessor2_0_multi

# pipe = Customized_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe = Customized_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
#                                                             torch_dtype=torch.float16,
#                                                             device_map="balanced")
pipe = TI2I_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                            torch_dtype=torch.float16,
                                                            device_map="balanced")


test_type_2_assignments = {
    "objimg_acttxt": ["obj","act"],
    "objimg_bgtxt": ["obj","bg"],
    "objimg_textxt": ["obj","tex"],
    "objtxt_actimg": ["obj","act"],
    "objtxt_bgimg": ["obj","bg"],
    "objtxt_teximg": ["obj","tex"],
}

test_type_2_image_assigment = {
    "objimg_acttxt": ["obj"],
    "objimg_bgtxt": ["obj"],
    "objimg_textxt": ["obj"],
    "objtxt_actimg": ["act"],
    "objtxt_bgimg": ["bg"],
    "objtxt_teximg": ["tex"],
}

def convert_to_raimg_prompt(source_prompt):
    """
        Parse source prompt to extract reference text for different components.
        Returns the cleaned main prompt and a dictionary of reference texts.
    """
    ref_txt_dict={}
    obj_match = re.findall(r"<obj>(.*?)</obj>", source_prompt)
    if obj_match:
        ref_txt_dict["obj"]=obj_match

    act_match = re.findall(r"<act>(.*?)</act>", source_prompt)
    if act_match:
        ref_txt_dict["act"]=act_match


    tex_match = re.findall(r"<tex>(.*?)</tex>", source_prompt)
    if tex_match:
        ref_txt_dict["tex"]=tex_match



    bg_match = re.findall(r"<bg>(.*?)</bg>", source_prompt)
    if bg_match:
        ref_txt_dict["bg"]=bg_match


    image_assignment = test_type_2_image_assigment[test_type]
    # sub_prompts=[]
    # for word in source_prompt.split():
    #     word = word.replace("<obj>", "</obj>").replace("<bg>", "</bg>").replace("<tex>", "</tex>").replace("<act>", "</act>")
    #     emu_prompt.append(word)
    return source_prompt.replace("<obj>","").replace("</obj>","").replace("<bg>","").replace("</bg>","").replace("<tex>","").replace("</tex>","").replace("<act>","").replace("</act>",""),ref_txt_dict



data_dir="./data/1_single_entry"
out_dir="./data/output_1_single_entry"


for test_type in  ["objimg_acttxt","objimg_textxt","objimg_bgtxt","objtxt_actimg","objtxt_bgimg","objtxt_teximg"][0:2]:
    if os.path.exists(f"{data_dir}/{test_type}"):
        print(test_type)
    refer_dir=f"ref_{test_type}"
    gen_dir=f"out_{test_type}"
    prompts_file=f"{test_type}.txt"

    refer_list=[]
    gen_list=[]
    source_prompt_list=[]
    obj_control_prompts_list=[]

    with open(f"{data_dir}/{prompts_file}", "r") as f:
        for line in f:
            source_prompt_list.append(line)
    if not os.path.exists(f"{out_dir}/{gen_dir}"):
        os.makedirs(f"{out_dir}/{gen_dir}")

    SKIP_NUM=3
    source_prompt_list = source_prompt_list[SKIP_NUM:]
    for index, source_prompt in tqdm(enumerate(source_prompt_list), total=len(source_prompt_list),
                                      desc=f"Generating entry: {test_type}, output: {out_dir}/{gen_dir}"):

        main_prompt, ref_txt_dict = convert_to_raimg_prompt(source_prompt)
        height=1024
        width=1024
        operator="concat"
        ref_img_path=f"{data_dir}/{refer_dir}/ref_{(index+SKIP_NUM):04d}.jpg"
        ref_image = PIL.Image.open(ref_img_path).convert("RGB").resize((height, width))
        images=[ref_image]
        prompt_a = main_prompt
        prompt_b = [main_prompt]

        layer_count = 0
        attn_processors = []
        def iter_net(net):
            global layer_count
            for child in net.children():
                if "Attention" in child.__class__.__name__:
                    child.processor = TI2I_JointAttnProcessor2_0_multi(contextual_replace=True, operator=operator,
                    wta_control_signal={"on":False},
                    ref_control_signal={"on":False})
                    attn_processors.append(child.processor)
                    layer_count += 1
                iter_net(child)

        iter_net(pipe.transformer)

        torch.manual_seed(42)
        pipe.enable_attention_slicing()

        switch_images = pipe.img2img_multi(
        images=images,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=3.5,
        strength=1,
        height=height,
        width=width,
        )
        switch_images[0].save(f"{out_dir}/{gen_dir}/{index:04d}.jpg")
        make_image_grid(switch_images, rows=1, cols=2).save(f"{out_dir}/{gen_dir}/comp_{index:04d}.jpg")
