"""
GRPO dataset: yields prompts, input images, and ground-truth answers
for rollout-based RL training. No token packing — each sample is
processed independently during rollout generation.
"""

import io
import re
import json
from typing import Optional
from PIL import Image, ImageFile, PngImagePlugin
import sys
sys.path.append("/data/b-bsachdeva/ThinkMorph")
from data.interleave_datasets.interleave_t2i_dataset import ParquetStandardIterableDataset
from data.data_utils import pil_img2rgb
from data.transforms import ImageTransform
from data.parquet_utils import get_parquet_data_paths
from data.dataset_info import DATASET_INFO

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


def extract_ground_truth(output_text_list) -> Optional[str]:
    """Extract the ground-truth answer from the last output text chunk.

    The final element of output_text_list typically contains
    <answer>…</answer>.  We extract the content inside.
    """
    if output_text_list is None or len(output_text_list) == 0:
        return None
    last_text = output_text_list[-1]
    matches = re.findall(r"<answer>(.*?)</answer>", last_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Fallback: return the full last text chunk (some tasks may not wrap in tags)
    return last_text.strip()


class GRPOIterableDataset(ParquetStandardIterableDataset):
    """
    Dataset for GRPO training.  Yields dicts with:
        - prompt (str): the question / instruction
        - input_image (PIL.Image): the problem image
        - ground_truth (str): the correct answer text
        - output_text_list (list[str]): full output texts for reference
        - image_list_raw (list[bytes]): raw image bytes for all images

    Unlike the SFT datasets this does NOT tokenise or pack samples.
    """

    def parse_row(self, row):
        instrs = row["instruction_list"]
        images = row["image_list"]
        outputs = row["output_text_list"]
        answer = row["answer"]
        # These come from parquet as numpy arrays; use len() instead of bool()
        if instrs is None or len(instrs) == 0:
            return {}
        if images is None or len(images) == 0:
            return {}
        if outputs is None or len(outputs) == 0:
            return {}

        # Extract ground-truth answer
        gt = extract_ground_truth(outputs)
        if gt is None or not gt:
            return {}

        # Decode input image
        try:
            input_image = pil_img2rgb(Image.open(io.BytesIO(images[0])))
        except Exception:
            return {}

        prompt = str(instrs[0]) if hasattr(instrs, '__len__') else str(instrs)

        return {
            "prompt": prompt,
            "input_image": input_image,
            "ground_truth": answer,
        }


def grpo_collate_fn(batch):
    """Custom collate that passes PIL images and strings through as-is."""
    out = {}
    for key in batch[0]:
        values = [d[key] for d in batch]
        out[key] = values
    return out


def build_grpo_dataset(
    dataset_config_meta: dict,
    tokenizer,
    local_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    data_status=None,
):
    """
    Build a GRPOIterableDataset from a YAML-loaded config dict.

    Args:
        dataset_config_meta: Parsed YAML with the same structure as
            data/configs/grpo_interleaved.yaml.
        tokenizer: Qwen2Tokenizer instance (needed by base class).
        local_rank, world_size, num_workers: distributed settings.

    Returns:
        GRPOIterableDataset instance.
    """
    datasets = []
    for group_name, group_cfg in dataset_config_meta.items():
        dataset_names = group_cfg["dataset_names"]
        img_args = group_cfg.get("image_transform_args", {})
        vit_args = group_cfg.get("vit_image_transform_args", {})
        num_used_data = group_cfg.get("num_used_data", [None] * len(dataset_names))

        transform = ImageTransform(
            max_image_size=img_args.get("max_image_size", 1024),
            min_image_size=img_args.get("min_image_size", 512),
            image_stride=img_args.get("image_stride", 16),
        )
        vit_transform = ImageTransform(
            max_image_size=vit_args.get("max_image_size", 518),
            min_image_size=vit_args.get("min_image_size", 224),
            image_stride=vit_args.get("image_stride", 14),
        )

        info = DATASET_INFO.get(group_name, {})
        data_dir_list = []
        parquet_info = {}
        num_used = []
        for i, ds_name in enumerate(dataset_names):
            ds_info = info.get(ds_name, {})
            data_dir_list.append(ds_info.get("data_dir", ""))
            n = num_used_data[i] if i < len(num_used_data) else ds_info.get("num_total_samples")
            num_used.append(n)
            pinfo_path = ds_info.get("parquet_info_path", "")
            if pinfo_path:
                try:
                    with open(pinfo_path, "r") as f:
                        parquet_info.update(json.load(f))
                except Exception:
                    pass

        ds = GRPOIterableDataset(
            dataset_name=group_name,
            transform=transform,
            tokenizer=tokenizer,
            vit_transform=vit_transform,
            data_dir_list=data_dir_list,
            num_used_data=num_used,
            parquet_info=parquet_info,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=data_status,
        )
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]
    # If multiple groups, chain them (round-robin would need a wrapper)
    from itertools import chain
    return chain(*datasets)

if __name__ == "__main__":
    # Quick test of dataset loading
    from modeling.qwen2 import Qwen2Tokenizer
    from data.data_utils import add_special_tokens
    tokenizer = Qwen2Tokenizer.from_pretrained("BAGEL-7B-MoT/")
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    dataset = build_grpo_dataset(
        dataset_config_meta={
            "thinkmorph_reasoning": {
                "dataset_names": ["Jigsaw_Assembly"],
                "num_used_data": [10],
            },
        },
        tokenizer=tokenizer,
        local_rank=0,
        world_size=1,
        num_workers=0,
    )
    for i, sample in enumerate(dataset):
        print(f"Sample {i}:")
        print("Prompt:", sample["prompt"])
        print("Ground truth:", sample["ground_truth"])
        print("Input image size:", sample["input_image"].size)
        if i >= 2:
            break
