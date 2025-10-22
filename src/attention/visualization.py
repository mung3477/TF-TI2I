from typing import Union, Literal, Optional, Tuple
from enum import Enum

import math

import torch
from einops import rearrange

AttnType = Literal["T2T", "T2I"]

class SavedAttnMapType(str, Enum):
		"""
		Enum for types of saved attention maps.
		"""
		T2I = "T2I"  # Text-to-Image attention maps
		I2T = "I2T"  # Image-to-Text attention maps

class ScalableAttentionProcessor:
	def __init__(self, height: int, width: int, save_attn_maps: Union[SavedAttnMapType, None] = None):
		self.height = height
		self.width = width
		self.num_pixels = height * width
		self.save_attn_maps = save_attn_maps

	def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
				is_causal=False, scale=None, enable_gqa=False) -> Tuple[torch.Tensor, torch.Tensor]:
		L, S = query.size(-2), key.size(-2)
		scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
		attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
		if is_causal:
				assert attn_mask is None
				temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
				attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
				attn_bias.to(query.dtype)

		if attn_mask is not None:
				if attn_mask.dtype == torch.bool:
						attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
				else:
						attn_bias += attn_mask

		if enable_gqa:
				key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
				value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

		attn_weight = query @ key.transpose(-2, -1) * scale_factor
		attn_weight += attn_bias
		attn_weight = torch.softmax(attn_weight, dim=-1)
		attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
		return attn_weight @ value, attn_weight

	def store_attn_map(self, attn_probs: torch.Tensor, text_length: int, attn_map_height: int, timestep: Optional[torch.Tensor] = None, attn_prompt: Optional[str] = None):
		if not hasattr(self, "attn_map"):
					self.attn_map = dict()
					self.attn_prompts = list()

		if self.save_attn_maps == SavedAttnMapType.I2T:
			self.I2T_attn_probs = attn_probs[:,:,text_length:,:text_length].detach()
			self.I2I_attn_probs = attn_probs[:,:,text_length:,text_length:].detach()

			# (1,24,4608,4608) -> (1,24,4096,512)
			attn_probs = attn_probs[:,:,text_length:,:text_length]
			attn_map = rearrange(
					attn_probs,
					'batch attn_head (height width) attn_dim -> batch attn_head height width attn_dim',
					height = attn_map_height
			) # (1,24,4096,512) -> (1,24,height,width,512)

			# Attention maps per prompt
			if attn_prompt is not None:
				self.attn_map[attn_prompt] = attn_map.detach().cpu()
				self.attn_prompts.append(attn_prompt)
			else:
				self.attn_map = attn_map.detach().cpu()

		elif self.save_attn_maps == SavedAttnMapType.T2I:
			# CAUTION: SD3 uses concat(I, T) while FLUX uses concat(T, I)
			self.T2T_attn_probs = attn_probs[:, :, -text_length:, -text_length:].detach()
			self.T2I_attn_probs = attn_probs[:, :, -text_length:, :-text_length].detach()

			# (1, 38, 4429, 4429) -> (1, 38, 333, 4096)
			attn_probs = attn_probs[:,:,-text_length:, :-text_length]
			attn_map = rearrange(
					attn_probs,
					'batch attn_head attn_dim (height width)  -> batch attn_head height width attn_dim',
					height = attn_map_height
			) # (1,24,512,4096) -> (1,24,height,width,512)

			# Attention maps per prompt
			if attn_prompt is not None:
				self.attn_map[attn_prompt] = attn_map.detach().cpu()
				self.attn_prompts.append(attn_prompt)
			else:
				self.attn_map = attn_map.detach().cpu()

		self.timestep = timestep[0].item() # TODO: int -> list
