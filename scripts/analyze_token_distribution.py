#!/usr/bin/env python3
"""
Analyze the distribution of num_tokens per sample across ThinkMorph datasets.

Reads parquet files directly, tokenizes text and computes image patch counts
using the same logic as the training data pipeline.

Usage:
    python analyze_token_distribution.py
    python analyze_token_distribution.py --max_samples 1000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import json
import argparse
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from modeling.qwen2 import Qwen2Tokenizer
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb
from tqdm import trange


def count_tokens_for_sample(row, tokenizer, img_transform, vit_transform):
    """Compute num_tokens + 2*len(sequence_plan) for one sample,
    matching UnifiedEditIterableDataset.parse_row + PackedDataset logic."""
    instrs = row["instruction_list"]
    images = row["image_list"]
    outputs = row["output_text_list"]

    num_tokens = 0
    num_plan_entries = 0

    # First image: need_vae=True, need_vit=True, need_loss=False
    img = pil_img2rgb(Image.open(io.BytesIO(images[0])))
    t = img_transform(img)
    h, w = t.shape[1:]
    num_tokens += w * h // img_transform.stride ** 2  # vae (no loss)
    num_plan_entries += 1  # vae_image
    vit_t = vit_transform(img)
    vh, vw = vit_t.shape[1:]
    num_tokens += vw * vh // vit_transform.stride ** 2  # vit
    num_plan_entries += 1  # vit_image

    # First instruction text (no loss)
    text_ids = tokenizer.encode(instrs[0])
    num_tokens += len(text_ids)
    num_plan_entries += 1  # text

    # Output texts + subsequent images
    for idx, out_txt in enumerate(outputs):
        text_ids = tokenizer.encode(out_txt)
        num_tokens += len(text_ids)
        num_plan_entries += 1  # text

        img_idx = idx + 1
        if img_idx < len(images):
            img = pil_img2rgb(Image.open(io.BytesIO(images[img_idx])))
            # need_loss=True: vae_image (loss=1)
            t = img_transform(img)
            h, w = t.shape[1:]
            num_tokens += w * h // img_transform.stride ** 2
            num_plan_entries += 1
            # need_vae=True: vae_image (loss=0)
            num_tokens += w * h // img_transform.stride ** 2
            num_plan_entries += 1
            # need_vit=True: vit_image
            vit_t = vit_transform(img)
            vh, vw = vit_t.shape[1:]
            num_tokens += vw * vh // vit_transform.stride ** 2
            num_plan_entries += 1

    # PackedDataset uses: num_tokens + 2 * len(sequence_plan)
    return num_tokens + 2 * num_plan_entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/b-bsachdeva/data/thinkmorph")
    parser.add_argument("--model_path", default="BAGEL-7B-MoT")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    img_transform = ImageTransform(image_stride=16, max_image_size=1024, min_image_size=512)
    vit_transform = ImageTransform(image_stride=14, max_image_size=518, min_image_size=224)

    datasets = ["Jigsaw_Assembly", "Spatial_Navigation", "Visual_Search", "Chart_Refocus"]
    token_counts = []
    dataset_labels = []
    total_processed = 0

    for ds_name in datasets:
        ds_dir = os.path.join(args.data_dir, ds_name)
        info_path = os.path.join(args.data_dir, "parquet_info", f"{ds_name}.json")
        with open(info_path) as f:
            parquet_info = json.load(f)

        print(f"Processing {ds_name} ...", flush=True)
        count = 0
        for pq_path in sorted(parquet_info.keys()):
            if not os.path.isfile(pq_path):
                print(f"  Warning: {pq_path} not found, skipping")
                continue
            table = pq.read_table(pq_path)
            for i in range(len(table)):
                row = {
                    "instruction_list": table["instruction_list"][i].as_py(),
                    "image_list": table["image_list"][i].as_py(),
                    "output_text_list": table["output_text_list"][i].as_py(),
                }
                try:
                    nt = count_tokens_for_sample(row, tokenizer, img_transform, vit_transform)
                    token_counts.append(nt)
                    dataset_labels.append(ds_name)
                except Exception as e:
                    print(f"  Warning: skipping sample {count} in {ds_name}: {e}")
                count += 1
                total_processed += 1
                if count % 500 == 0:
                    print(f"  {ds_name}: {count} samples", flush=True)
                if args.max_samples and total_processed >= args.max_samples:
                    break
            if args.max_samples and total_processed >= args.max_samples:
                break
        print(f"  {ds_name}: {count} samples done", flush=True)
        if args.max_samples and total_processed >= args.max_samples:
            break

    if not token_counts:
        print("No samples found!")
        return

    # ── Statistics ─────────────────────────────────────────────────────────
    arr = np.array(token_counts)
    print("\n" + "=" * 60)
    print(f"{'OVERALL STATISTICS':^60}")
    print("=" * 60)
    print(f"Total samples: {len(arr):,}")
    print(f"Min tokens:    {arr.min():>8,}")
    print(f"Max tokens:    {arr.max():>8,}")
    print(f"Mean:          {arr.mean():>8,.1f}")
    print(f"Median:        {np.median(arr):>8,.1f}")
    print(f"Std:           {arr.std():>8,.1f}")
    print()
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:<2}:  {np.percentile(arr, p):>8,.0f}")

    # ── Histogram ─────────────────────────────────────────────────────────
    bins = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, float("inf")]
    labels = [
        "0-1K", "1K-2K", "2K-4K", "4K-8K",
        "8K-16K", "16K-32K", "32K-64K", "64K+",
    ]
    counts_per_bin = [0] * len(labels)
    for v in arr:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts_per_bin[i] += 1
                break

    print("\n" + "=" * 60)
    print(f"{'TOKEN COUNT DISTRIBUTION':^60}")
    print("=" * 60)
    print(f"{'Range':<12} {'Count':>8} {'Pct':>7} {'Cumul':>7}  Bar")
    print("-" * 60)
    max_count = max(counts_per_bin) if counts_per_bin else 1
    cumul = 0.0
    for label, cnt in zip(labels, counts_per_bin):
        pct = 100.0 * cnt / len(arr)
        cumul += pct
        bar = "█" * int(40 * cnt / max_count)
        print(f"{label:<12} {cnt:>8,} {pct:>6.1f}% {cumul:>6.1f}%  {bar}")

    # ── Per-dataset breakdown ─────────────────────────────────────────────
    unique_ds = sorted(set(dataset_labels))
    if len(unique_ds) > 1:
        print("\n" + "=" * 60)
        print(f"{'PER-DATASET BREAKDOWN':^60}")
        print("=" * 60)
        for ds in unique_ds:
            ds_arr = np.array([t for t, l in zip(token_counts, dataset_labels) if l == ds])
            print(f"\n  {ds} (n={len(ds_arr):,})")
            print(f"    min={ds_arr.min():>6,}  max={ds_arr.max():>6,}  "
                  f"mean={ds_arr.mean():>8,.1f}  median={np.median(ds_arr):>8,.1f}")
            # mini histogram
            ds_bins = [0] * len(labels)
            for v in ds_arr:
                for i in range(len(bins) - 1):
                    if bins[i] <= v < bins[i + 1]:
                        ds_bins[i] += 1
                        break
            for label, cnt in zip(labels, ds_bins):
                if cnt > 0:
                    pct = 100.0 * cnt / len(ds_arr)
                    print(f"    {label:<12} {cnt:>6,} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
