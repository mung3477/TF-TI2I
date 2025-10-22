import os
os.environ["HF_HOME"] = "/root/Desktop/workspace/yujin/woosung/.hub"

from attention_map_diffusers import (
	init_pipeline, save_attention_maps, attn_maps
)

import PIL
from diffusers.utils import make_image_grid
import torch
from tqdm import tqdm

from src.customized_pipe import TI2I_StableDiffusion3Pipeline
from src.attn_processor import TI2I_JointAttnProcessor2_0_multi
from src.util.argparse import parse_args
from src.util.save import get_image_name, get_attn_map_dir


# pipe = Customized_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe = Customized_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
#                                                             torch_dtype=torch.float16,
#                                                             device_map="balanced")
pipe = TI2I_StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large",
                                                            torch_dtype=torch.float16,
                                                            device_map="balanced")



from PIL import Image
operator = "concat"
from tqdm import tqdm
import re
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


    # sub_prompts=[]
    # for word in source_prompt.split():
    #     word = word.replace("<obj>", "</obj>").replace("<bg>", "</bg>").replace("<tex>", "</tex>").replace("<act>", "</act>")
    #     emu_prompt.append(word)
    return source_prompt.replace("<obj>","").replace("</obj>","").replace("<bg>","").replace("</bg>","").replace("<tex>","").replace("</tex>","").replace("<act>","").replace("</act>",""),ref_txt_dict



data_dir="./data/4_quad_entry"
out_dir="./data/output_4_quad_entry"

args = parse_args()


# "objimg_acttxt",
for test_type in  ["case_1"]:
    if os.path.exists(f"{data_dir}/{test_type}"):
        print(test_type)
    refer_dir="refers"
    gen_dir=f"wta_refer"
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
    for index, source_prompt in tqdm(enumerate(source_prompt_list), total=len(source_prompt_list),
                                      desc=f"Generating entry: {test_type}, output: {out_dir}/{gen_dir}"):

        main_prompt, ref_txt_dict = convert_to_raimg_prompt(source_prompt)
        height=1024
        width=1024
        operator="concat"
        obj_prompt = ref_txt_dict["obj"][0]
        tex_prompt = ref_txt_dict["tex"][0]
        act_prompt = ref_txt_dict["act"][0]
        bg_prompt = ref_txt_dict["bg"][0]

        main_prompt=f"{obj_prompt} with texture of {tex_prompt} doing {act_prompt} in {bg_prompt}"
        sub_prompts=[obj_prompt, tex_prompt, act_prompt, bg_prompt]
        ##### @attention_map_visualization #####
        if bool(args.save_attn_maps):
            for prompt in sub_prompts:
                attn_maps[prompt] = attn_maps.get(prompt, dict())
        #############################################

        refer_images=[]
        torch.manual_seed(args.seed)
        for i_sbp, sbp in enumerate(sub_prompts):
            ref_img = Image.open(f"{data_dir}/refers/{index}_{i_sbp}.png")
            refer_images.append(ref_img)

        torch.manual_seed(args.seed)
        pipe.enable_attention_slicing()

        layer_count = 0
        attn_processors = []
        def iter_net(net):
            global layer_count
            for child in net.children():
                if "Attention" in child.__class__.__name__:
                    child.processor = TI2I_JointAttnProcessor2_0_multi(
                        layer=layer_count,
                        contextual_replace=True,
                        wta_control_signal={
                            "on":True,
                            "hyper_parameter":{
                                "wta_weight":[1, 1, 1,1],
                                "cross2ref":True,
                                "wta_shift":[0, 0, 0,0],
                                "wta_cross":False
                            },
                            "debug":False
                        },
                        ref_control_signal={
                            "on":True,
                            "ref_idxs":[0,1,2,3],
                            "ref_prompts": sub_prompts,
                            "control_type":"main_context",
                            "debug":False,
                            "control_layers":[i for i in range(25,40)],
                            "hyper_parameter":{}
                        },
                        width=width // 16,
                        height=height // 16,
                    )
                    ##### @attention_map_visualization #####
                    if bool(args.save_attn_maps):
                        child.processor.save_attn_maps = args.save_attn_maps
                    attn_processors.append(child.processor)
                    layer_count += 1
                iter_net(child)

        iter_net(pipe.transformer)
        print(f"Added attention maps storing mode to {len(attn_processors)} modules.")

        torch.manual_seed(0)
        pipe.enable_attention_slicing()
        pipe = init_pipeline(pipe)

        switch_images = pipe.img2img_multi(
            images=refer_images,
            prompt_a=main_prompt,
            prompt_b=sub_prompts,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=5,
            strength=1,
            height=1024,
            width=1024,
        )
        image_name = f"{index:04d}-{get_image_name(args)}"
        grid_img=make_image_grid(switch_images, rows=1, cols=len(switch_images))
        switch_images[0].save(f"{out_dir}/{gen_dir}/{image_name}.jpg")
        grid_img.save(f"{out_dir}/{gen_dir}/comp_{image_name}.jpg")

        ##### @attention_map_visualization #####
        if bool(args.save_attn_maps):
            for sub_prompt in sub_prompts:
                ref_prompt_attn_map = attn_maps[sub_prompt]
                attn_base_dir = get_attn_map_dir(out_dir, gen_dir, image_name, args.save_attn_maps) + f"/{sub_prompt}"
                save_attention_maps(ref_prompt_attn_map, pipe.tokenizer, sub_prompt, base_dir=attn_base_dir, unconditional=False)
        #############################################
