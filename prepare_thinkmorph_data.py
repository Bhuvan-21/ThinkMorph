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
  output_text_list  = [resoning_thought_0, resoning_thought_1]

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

    if not question or (not thought_0 and not thought_1):
        return None

    return {
        "instruction_list": [question],
        "image_list":        [prob_bytes, reas_bytes],
        "output_text_list":  [thought_0, thought_1],
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
    parser.add_argument("--output_dir",     default="/home/b-bsachdeva/data")
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

    if not args.no_update_dataset_info:
        _update_dataset_info(all_info)
        _write_thinkmorph_yaml(all_info)

    print("\n✓ Done.")


def _update_dataset_info(all_info):
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    info_py_path = os.path.join(script_dir, "data", "dataset_info.py")

    with open(info_py_path, "r") as f:
        src = f.read()

    lines = [
        "\n\n# ThinkMorph interleaved reasoning datasets (auto-generated by prepare_thinkmorph_data.py)",
        "THINKMORPH_DATASET_INFO = {",
        "    'thinkmorph_reasoning': {",
    ]
    for name, meta in all_info.items():
        lines += [
            f"        {repr(name)}: {{",
            f"            'data_dir': {repr(meta['data_dir'])},",
            f"            'num_files': {meta['num_files']},",
            f"            'num_total_samples': {meta['num_total_samples']},",
            f"            'parquet_info_path': {repr(meta['parquet_info_path'])},",
            "        },",
        ]
    lines += ["    },", "}", "\nDATASET_INFO.update(THINKMORPH_DATASET_INFO)"]
    block = "\n".join(lines)

    marker = "# ThinkMorph interleaved reasoning datasets (auto-generated"
    if marker in src:
        src = re.sub(marker + r".*", block.lstrip("\n"), src, flags=re.DOTALL)
    else:
        src = src + block

    with open(info_py_path, "w") as f:
        f.write(src)
    print(f"Updated {info_py_path}")


def _write_thinkmorph_yaml(all_info):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path  = os.path.join(script_dir, "data", "configs", "thinkmorph_reasoning.yaml")

    names    = list(all_info.keys())
    num_used = [meta["num_total_samples"] for meta in all_info.values()]

    lines = ["thinkmorph_reasoning:", "  dataset_names:"]
    for n in names:
        lines.append(f"  - {n}")
    lines += [
        "  image_transform_args:",
        "    image_stride: 16",
        "    max_image_size: 1024",
        "    min_image_size: 512",
        "  vit_image_transform_args:",
        "    image_stride: 14",
        "    max_image_size: 518",
        "    min_image_size: 224",
        "  is_mandatory: true",
        "  num_used_data:",
    ]
    for n in num_used:
        lines.append(f"  - {n}")
    lines += ["  weight: 1", ""]

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {yaml_path}")
    print(f"\nNext: DATASET_CONFIG=data/configs/thinkmorph_reasoning.yaml bash scripts/train_lora.sh")


if __name__ == "__main__":
    main()
