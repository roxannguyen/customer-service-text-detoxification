"""
generate_web_data_from_up.py

Converts processed evaluation datasets from data_processed/ into English-only JSON assets
for the Next.js human annotation platform (both local and Vercel environments):
- task_a/task_a_web_annotation_240.csv -> website/src/data/task_a_data.json (240 items)
- task_b/task_b_web_annotation_234_pairs.csv -> website/src/data/task_b_data.json (234 unique pairs)
- task_b/task_b_pair_mapping_300.csv -> website/src/data/task_b_mapping.json (300 model output mappings)
"""

import json
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_A_DIR = SCRIPT_DIR.parent / "task_a"
TASK_B_DIR = SCRIPT_DIR.parent / "task_b"

# Target output directories for local and deployment
OUT_DIRS = [
    SCRIPT_DIR.parent.parent / "website" / "src" / "data",
    SCRIPT_DIR.parent.parent.parent / "website" / "Website_coding" / "src" / "data"
]

for out_dir in OUT_DIRS:
    out_dir.mkdir(parents=True, exist_ok=True)

# 1. TASK A: Convert 240 items to task_a_data.json (English only)
csv_a = TASK_A_DIR / "task_a_web_annotation_240.csv"
assert csv_a.exists(), f"Task A source file not found: {csv_a}"
df_a = pd.read_csv(csv_a)
task_a_items = []
for idx, row in df_a.iterrows():
    task_a_items.append({
        "item_id": idx + 1,
        "sample_id": str(row["id"]),
        "text": str(row["text"]),
        "original_label": str(row.get("label", "")),
        "subtype": str(row.get("subtype", "")),
        "topic_type": str(row.get("topic_type", "")),
        "scenario": str(row.get("scenario", "")),
        "industry": str(row.get("industry", "")),
        "register": str(row.get("register", "")),
        "detector_score": float(row.get("score", 0.0)) if pd.notnull(row.get("score")) else None,
        "band": str(row.get("band", "")),
        "off_target": bool(row.get("off_target", False)),
    })

for out_dir in OUT_DIRS:
    with open(out_dir / "task_a_data.json", "w", encoding="utf-8") as f:
        json.dump(task_a_items, f, ensure_ascii=False, indent=2)
print(f"Task A: Exported {len(task_a_items)} English items to task_a_data.json in all environments")

# 2. TASK B: Convert 234 deduplicated pairs to task_b_data.json (English only)
csv_b_234 = TASK_B_DIR / "task_b_web_annotation_234_pairs.csv"
assert csv_b_234.exists(), f"Task B 234 pairs source file not found: {csv_b_234}"
df_b_234 = pd.read_csv(csv_b_234)
task_b_items = []
for idx, row in df_b_234.iterrows():
    task_b_items.append({
        "item_number": int(row.get("item_number", idx + 1)),
        "sample_id": str(row["pair_id"]),
        "pair_id": str(row["pair_id"]),
        "input_id": int(row["input_id"]),
        "input_text": str(row["input_text"]),
        "output_text": str(row["output_text"]),
        "system_id": str(row.get("systems", "")),
        "systems": str(row.get("systems", "")),
        "num_systems": int(row.get("num_systems", 1)),
        "seed": None,
        "band": str(row.get("band", "")),
        "subtype": str(row.get("subtype", "")),
        "original_label": str(row.get("original_label", "")),
    })

for out_dir in OUT_DIRS:
    with open(out_dir / "task_b_data.json", "w", encoding="utf-8") as f:
        json.dump(task_b_items, f, ensure_ascii=False, indent=2)
print(f"Task B: Exported {len(task_b_items)} English unique pairs to task_b_data.json in all environments")

# 3. TASK B MAPPING: Convert 300 model output mappings to task_b_mapping.json
csv_mapping = TASK_B_DIR / "task_b_pair_mapping_300.csv"
assert csv_mapping.exists(), f"Task B mapping source file not found: {csv_mapping}"
df_mapping = pd.read_csv(csv_mapping)
mapping_items = []
for idx, row in df_mapping.iterrows():
    mapping_items.append({
        "system_sample_id": str(row["system_sample_id"]),
        "pair_id": str(row["pair_id"]),
        "input_id": int(row["input_id"]),
        "system_id": str(row["system_id"]),
        "seed": int(row["seed"]) if pd.notnull(row.get("seed")) else None,
        "output_text": str(row["output_text"]),
        "band": str(row.get("band", "")),
        "subtype": str(row.get("subtype", "")),
        "original_label": str(row.get("original_label", "")),
    })

for out_dir in OUT_DIRS:
    with open(out_dir / "task_b_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_items, f, ensure_ascii=False, indent=2)
print(f"Task B Mapping: Exported {len(mapping_items)} mapping rows to task_b_mapping.json in all environments")
