#!/usr/bin/env python3
"""
Split the prepared ThinkMorph parquet shards into disjoint SFT / GRPO subsets
by symlinking shards into per-task `sft/` and `grpo/` subdirectories.

Stratified by task to preserve task diversity in both phases. The first K
shards (lexicographic) of each task go to SFT, the rest to GRPO.

After running, register the new entries in data/dataset_info.py and point the
YAMLs at <task>_sft / <task>_grpo.

Usage:
    python split_thinkmorph_data.py [--root /workspace/data/thinkmorph]
"""

import argparse
import json
import os

import pyarrow.parquet as pq


# task -> (#shards for SFT, total shards)
SPLIT_PLAN = {
    "Jigsaw_Assembly":    (5, 12),
    "Spatial_Navigation": (5, 12),
    "Visual_Search":      (6, 14),
    "Chart_Refocus":      (5, 12),
}


def split_task(root: str, info_dir: str, task: str, n_sft: int, n_total: int):
    src_dir = os.path.join(root, task)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"missing task dir: {src_dir}")

    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".parquet"))
    if len(shards) != n_total:
        raise RuntimeError(
            f"{task}: expected {n_total} shards, found {len(shards)} — "
            f"adjust SPLIT_PLAN if your data layout differs."
        )

    splits = {"sft": shards[:n_sft], "grpo": shards[n_sft:]}
    summary = {}

    for split_name, shard_list in splits.items():
        out_dir = os.path.join(src_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)

        parquet_info = {}
        n_rows = 0
        for shard in shard_list:
            src = os.path.join(src_dir, shard)
            dst = os.path.join(out_dir, shard)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)

            pf = pq.ParquetFile(dst)
            parquet_info[dst] = {"num_row_groups": pf.metadata.num_row_groups}
            n_rows += pf.metadata.num_rows

        info_path = os.path.join(info_dir, f"{task}_{split_name}.json")
        with open(info_path, "w") as f:
            json.dump(parquet_info, f, indent=2)

        summary[split_name] = {
            "data_dir":          out_dir,
            "num_files":         len(shard_list),
            "num_total_samples": n_rows,
            "parquet_info_path": info_path,
        }
        print(
            f"  {task:20s} {split_name:4s}: {len(shard_list):2d} shards, "
            f"{n_rows:5d} rows → {out_dir}"
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/data/thinkmorph")
    args = parser.parse_args()

    info_dir = os.path.join(args.root, "parquet_info")
    os.makedirs(info_dir, exist_ok=True)

    print(f"Splitting under {args.root}")
    grand = {"sft": 0, "grpo": 0}
    snippets = {}
    for task, (n_sft, n_total) in SPLIT_PLAN.items():
        s = split_task(args.root, info_dir, task, n_sft, n_total)
        snippets[task] = s
        grand["sft"]  += s["sft"]["num_total_samples"]
        grand["grpo"] += s["grpo"]["num_total_samples"]

    print(f"\nTotal SFT samples : {grand['sft']:>6d}")
    print(f"Total GRPO samples: {grand['grpo']:>6d}")

    print("\nPaste these into THINKMORPH_DATASET_INFO['thinkmorph_reasoning'] "
          "in data/dataset_info.py:\n")
    for task, s in snippets.items():
        for split, info in s.items():
            print(f"    '{task}_{split}': {{")
            for k, v in info.items():
                print(f"        '{k}': {v!r},")
            print("    },")


if __name__ == "__main__":
    main()
