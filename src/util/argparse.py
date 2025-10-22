import argparse
from typing import Union, Optional

from src.attention.visualization import SavedAttnMapType

def parse_saved_attn_map_type(s: Optional[str]) -> Union[SavedAttnMapType, None]:
		if s not in (None, "T2I", "I2T"):
				raise ValueError(f"Invalid SavedAttnMapType: {s}")
		return SavedAttnMapType(s) if s is not None else None

def parse_args():
		parser = argparse.ArgumentParser()
		parser.add_argument('--save_attn_maps', type=str, default=None, help="None | T2I | I2T")
		parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")

		args = parser.parse_args()
		args.save_attn_maps = parse_saved_attn_map_type(args.save_attn_maps)

		return args
