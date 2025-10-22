import os
from src.attention.visualization import SavedAttnMapType

def get_image_name(args):
        name = f"seed-{args.seed}"

        return name

def get_attn_map_dir(base_dir, target_prompt, image_name, save_attn: SavedAttnMapType):
  attn_base_dir = f"{base_dir}/{target_prompt}/{image_name}"

  if bool(save_attn):
    attn_base_dir += f"-{save_attn.value}"

  if not os.path.exists(attn_base_dir):
    os.makedirs(attn_base_dir)

  return attn_base_dir
