# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Download and convert all four ThinkMorph HuggingFace datasets into the parquet
format expected by UnifiedEditIterableDataset.

Uses huggingface_hub + pyarrow directly (no `datasets` lib) so it works with
the pyarrow==11 already installed in the thinkmorph conda environment.

Usage
─────
  python prepare_thinkmorph_data.py --output_dir /home/b-bsachdeva/data

HF parquet schema (all four tasks share this layout)
─────────────────────────────────────────────────────
  problem_image_0    : {bytes, path}
  question           : str
  resoning_thought_0 : str
  reasoning_image_0  : {bytes, path}
  resoning_thought_1 : str

Mapping → UnifiedEditIterableDataset parquet columns
──────────────────────────────────────────────────────
  instruction_list  = [question]
  image_list        = [problem_image_0.bytes, reasoning_image_0.bytes]
  output_text_list  = [
      f"<think>{resoning_thought_0}</think><image_start>",
      f"<image_end><think>{resoning_thought_1}</think><answer>{answer}</answer>",
  ]
  answer            = answer  (raw GT for GRPO reward calculation)

Training sequence per sample:
  [INPUT image (no loss)] → [question (no loss)] →
  [thought_0 (loss)] → [reasoning image (loss)] → [thought_1 (loss)]
"""

import argparse
import io
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_download
from PIL import Image
from tqdm import tqdm

HF_DATASETS = [
    "ThinkMorph/Jigsaw_Assembly",
    "ThinkMorph/Spatial_Navigation",
    "ThinkMorph/Visual_Search",
    "ThinkMorph/Chart_Refocus",
]

ROWS_PER_SHARD = 500

PARQUET_SCHEMA = pa.schema([
    pa.field("instruction_list", pa.list_(pa.string())),
    pa.field("image_list",       pa.list_(pa.binary())),
    pa.field("output_text_list", pa.list_(pa.string())),
    pa.field("answer",           pa.string()),
])


def dataset_name(hf_repo):
    return hf_repo.split("/")[-1]


def img_field_to_bytes(field):
    """HF stores images as {'bytes': b'...', 'path': '...'}. Re-encode to JPEG."""
    if field is None:
        return None
    raw = field.get("bytes") if isinstance(field, dict) else None
    if not raw:
        return None
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception as e:
        print(f"    [skip] image decode failed: {e}")
        return None


def convert_row(row):
    prob_bytes = img_field_to_bytes(row.get("problem_image_0"))
    reas_bytes = img_field_to_bytes(row.get("reasoning_image_0"))
    if prob_bytes is None or reas_bytes is None:
        return None

    question  = str(row.get("question", "")).strip()
    thought_0 = str(row.get("resoning_thought_0", "")).strip()
    thought_1 = str(row.get("resoning_thought_1", "")).strip()
    answer    = str(row.get("answer", "")).strip()

    if not question or not thought_0 or not thought_1 or not answer:
        return None

    return {
        "instruction_list": [question],
        "image_list":        [prob_bytes, reas_bytes],
        "output_text_list":  [
            f"<think>{thought_0}</think><image_start>",
            f"<image_end><think>{thought_1}</think><answer>{answer}</answer>",
        ],
        "answer": answer,
    }


def write_shards(rows, out_dir, rows_per_shard=500):
    os.makedirs(out_dir, exist_ok=True)
    parquet_info = {}
    for shard_idx, start in enumerate(range(0, len(rows), rows_per_shard)):
        chunk = rows[start: start + rows_per_shard]
        out_path = os.path.join(out_dir, f"part-{shard_idx:05d}.parquet")
        table = pa.Table.from_pylist(chunk, schema=PARQUET_SCHEMA)
        pq.write_table(table, out_path, row_group_size=100)
        pf = pq.ParquetFile(out_path)
        parquet_info[out_path] = {"num_row_groups": pf.metadata.num_row_groups}
    print(f"  → {len(rows)} samples in {len(parquet_info)} shard(s) → {out_dir}")
    return parquet_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",     default="/workspace/data")
    parser.add_argument("--datasets",       nargs="+", default=HF_DATASETS)
    parser.add_argument("--rows_per_shard", type=int,  default=500)
    parser.add_argument("--no_update_dataset_info", action="store_true")
    args = parser.parse_args()

    rows_per_shard = args.rows_per_shard

    base_dir = os.path.join(args.output_dir, "thinkmorph")
    info_dir = os.path.join(base_dir, "parquet_info")
    os.makedirs(info_dir, exist_ok=True)

    all_info = {}

    for hf_repo in args.datasets:
        name = dataset_name(hf_repo)
        print(f"\n{'─'*60}")
        print(f"Processing {hf_repo}")

        hf_files = sorted(
            f for f in list_repo_files(hf_repo, repo_type="dataset")
            if f.startswith("data/") and f.endswith(".parquet")
        )
        print(f"  {len(hf_files)} source shard(s) on HuggingFace.")

        rows = []
        n_total = 0

        for hf_file in hf_files:
            print(f"  Downloading {hf_file} …")
            local_path = hf_hub_download(hf_repo, hf_file, repo_type="dataset")
            pf = pq.ParquetFile(local_path)
            for rg_idx in tqdm(range(pf.metadata.num_row_groups),
                               desc=f"    row-groups", leave=False):
                df = pf.read_row_group(rg_idx).to_pandas()
                for _, row in df.iterrows():
                    n_total += 1
                    converted = convert_row(row.to_dict())
                    if converted is not None:
                        rows.append(converted)

        print(f"  {len(rows)} / {n_total} rows converted.")

        shard_dir    = os.path.join(base_dir, name)
        parquet_info = write_shards(rows, shard_dir, rows_per_shard)

        info_path = os.path.join(info_dir, f"{name}.json")
        with open(info_path, "w") as f:
            json.dump(parquet_info, f, indent=2)
        print(f"  Parquet info → {info_path}")

        all_info[name] = {
            "data_dir":          shard_dir,
            "num_files":         len(parquet_info),
            "num_total_samples": len(rows),
            "parquet_info_path": info_path,
        }

    print("\n✓ Done.")

if __name__ == "__main__":
    main()
