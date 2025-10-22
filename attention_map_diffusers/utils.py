import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import cv2
import numpy as np
from PIL import Image

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)

from .modules import *

def calc_self_cross_attention_ratios(self_scores: torch.Tensor, cross_scores: torch.Tensor):
	# Calculate mean scores to avoid division by zero
	self_mean = torch.mean(self_scores).item() if self_scores is not None else 0.0
	cross_mean = torch.mean(cross_scores).item() if cross_scores is not None else 0.0

	# Calculate ratio (handle division by zero)
	if cross_mean > 0:
		ratio = self_mean / cross_mean
	else:
		ratio = float('inf') if self_mean > 0 else 0.0

	return self_mean, cross_mean, ratio

def hook_function(name):
    def forward_hook(module, input, output):
        ################# CUSTOM #################
        if hasattr(module.processor, 'T2T_attn_probs') and hasattr(module.processor, 'T2I_attn_probs'):
            t2t_scores = module.processor.T2T_attn_probs
            t2i_scores = module.processor.T2I_attn_probs
            t2t_mean, t2i_mean, ratio = calc_self_cross_attention_ratios(t2t_scores, t2i_scores)

            ratios_data.append({
                'module_name': name,
                'T2T_mean_score': t2t_mean,
                'T2I_mean_score': t2i_mean,
                'T2T_T2I_ratio': ratio
            })

            del module.processor.T2T_attn_probs
            del module.processor.T2I_attn_probs

        elif hasattr(module.processor, 'I2T_attn_probs') and hasattr(module.processor, 'I2I_attn_probs'):
            i2i_scores = module.processor.I2I_attn_probs
            i2t_scores = module.processor.I2T_attn_probs
            i2i_mean, i2t_mean, ratio = calc_self_cross_attention_ratios(i2i_scores, i2t_scores)

            ratios_data.append({
                'module_name': name,
                'I2I_mean_score': i2i_mean,
                'I2T_mean_score': i2t_mean,
                'I2I_I2T_ratio': ratio
            })

            del module.processor.I2I_attn_probs
            del module.processor.I2T_attn_probs
        ################# CUSTOM #################

        if hasattr(module.processor, "attn_map") and len(module.processor.attn_map.keys()) > 0:
            timestep = module.processor.timestep
            attn_prompts = module.processor.attn_prompts

            # Store attention maps for multiple prompts
            if attn_prompts is not None:
                for prompt in attn_prompts:
                    attn_maps[prompt] = attn_maps.get(prompt, dict())
                    attn_maps[prompt][timestep] = attn_maps[prompt].get(timestep, dict())
                    attn_maps[prompt][timestep][name] = module.processor.attn_map[prompt]

            else:
                attn_maps[timestep] = attn_maps.get(timestep, dict())
                attn_maps[timestep][name] = module.processor.attn_map

            del module.processor.attn_map
            del module.processor.timestep
            del module.processor.attn_prompts

    return forward_hook


def register_cross_attention_hook(model, hook_function, target_name):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        if isinstance(module.processor, AttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, AttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, LoRAAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, JointAttnProcessor2_0):
            module.processor.store_attn_map = True
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            module.processor.store_attn_map = True

        hook = module.register_forward_hook(hook_function(name))

    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():

        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)

        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)

        replace_call_method_for_unet(layer)

    return model


# TODO: implement
# def replace_call_method_for_sana(model):
#     if model.__class__.__name__ == 'SanaTransformer2DModel':
#         from diffusers.models.transformers import SanaTransformer2DModel
#         model.forward = SanaTransformer2DModelForward.__get__(model, SanaTransformer2DModel)

#     for name, layer in model.named_children():

#         if layer.__class__.__name__ == 'SanaTransformerBlock':
#             from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
#             layer.forward = SanaTransformerBlockForward.__get__(layer, SanaTransformerBlock)

#         replace_call_method_for_sana(layer)

#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)

        replace_call_method_for_sd3(layer)

    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():

        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)

        elif layer.__class__.__name__ == 'FluxSingleTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
            layer.forward = FluxSingleTransformerBlockForward.__get__(layer, FluxSingleTransformerBlock)

        replace_call_method_for_flux(layer)

    return model


def init_pipeline(pipeline):
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)

        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn')
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2')
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)

        # Convert tensor to numpy and apply JET colormap
        attention_array = a.to(torch.float32).cpu().numpy()

        # Normalize to 0-1 range
        attention_array = (attention_array - attention_array.min()) / (attention_array.max() - attention_array.min() + 1e-8)

        # Convert to PIL Image (remove alpha channel and scale to 0-255)
        gray_attention_rgb = (attention_array * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(gray_attention_rgb, cv2.COLORMAP_JET)
        pil_image = Image.fromarray(colormap)

        pil_image.save(os.path.join(batch_dir, f'{i}-{token}.png'))


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()

    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]

    os.makedirs(base_dir, exist_ok=True)

    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0

    for timestep, layers in tqdm(attn_maps.items(), desc=f"Saving attention maps for {prompts}"):
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)

        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)

            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]

            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1

            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch}')
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)

    total_attn_map /= total_attn_map_number
    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f'batch-{batch}')
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)
