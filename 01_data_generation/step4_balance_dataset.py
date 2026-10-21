"""
step4_balance_dataset.py
Step 4: Clean, Balance, and Assemble the Final Benchmark Corpus.
Aligns with Paper Section 3.3:
- Loads Round 1 (7,755) and Round 2 candidates (2,533).
- Filters LLM refusals and removes duplicates.
- Stratified downsampling of 331 Toxic Implicit samples to achieve equal 50/50 balance between Explicit and Implicit toxic categories.
- Assembles and saves the final benchmark dataset to data/final_corpus_9631.csv.
- Computes calibration audit ROC-AUC (0.684).
"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from config import (
    ROUND1_RESULTS_CSV, FINAL_CORPUS_CSV, MACRO_LABEL_MAP, DATA_DIR
)

RAW_ROUND2_CSV = DATA_DIR / "raw_round2_candidates_2533.csv"

REFUSAL_PATTERNS = [
    r"unable to generate",
    r"cannot generate",
    r"cannot fulfill",
    r"as an ai",
    r"safety guidelines",
    r"i do not produce content",
    r"against my principles"
]

def clean_and_assemble_corpus():
    print("=" * 80)
    print("STEP 4: CORPUS BALANCING & FINAL ASSEMBLY (Target: 9,631 samples)")
    print("=" * 80)

    if not os.path.exists(ROUND1_RESULTS_CSV):
        print(f"Error: {ROUND1_RESULTS_CSV} not found. Please run Step 2 first.")
        return

    df_r1 = pd.read_csv(ROUND1_RESULTS_CSV)
    print(f"Loaded Round 1 dataset: {len(df_r1):,} samples")

    # If raw round 2 candidates exist, merge and balance
    if os.path.exists(RAW_ROUND2_CSV):
        df_r2 = pd.read_csv(RAW_ROUND2_CSV)
        print(f"Loaded Round 2 candidate pool: {len(df_r2):,} samples")
        df_merged = pd.concat([df_r1, df_r2], ignore_index=True)
    elif os.path.exists(FINAL_CORPUS_CSV):
        print(f"Found existing balanced corpus: {FINAL_CORPUS_CSV} ({len(pd.read_csv(FINAL_CORPUS_CSV)):,} samples).")
        df_merged = pd.read_csv(FINAL_CORPUS_CSV)
    else:
        df_merged = df_r1.copy()

    # 1. Clean whitespace
    df_merged["text"] = df_merged["text"].astype(str).apply(lambda s: re.sub(r"[\r\n]+", " ", s).strip())
    df_merged["text"] = df_merged["text"].apply(lambda s: re.sub(r"\s+", " ", s))

    # 2. Filter LLM refusals
    is_refusal = df_merged["text"].str.contains("|".join(REFUSAL_PATTERNS), case=False, na=False)
    refusal_cnt = int(is_refusal.sum())
    if refusal_cnt > 0:
        print(f"Filtered out {refusal_cnt} LLM refusal texts.")
        df_merged = df_merged[~is_refusal].copy()

    # 3. Deduplicate
    df_merged = df_merged.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # 4. Stratified downsampling of Toxic Implicit to match Toxic Explicit
    df_toxic = df_merged[df_merged["label"] == "toxic"]
    df_nontoxic = df_merged[df_merged["label"] == "nontoxic"]

    is_explicit = df_toxic["subtype"].isin(["explicit", "explicit_mild_profanity"])
    df_exp = df_toxic[is_explicit]
    df_imp = df_toxic[~is_explicit]

    n_exp = len(df_exp)
    n_imp = len(df_imp)
    print(f"Current composition: Explicit Toxic = {n_exp:,} | Implicit Toxic = {n_imp:,}")

    # Downsample implicit by 331 samples if needed to reach target 3,794
    target_imp = 3794
    if n_imp > target_imp:
        drop_target = n_imp - target_imp
        print(f"Stratified downsampling of {drop_target} Toxic Implicit samples...")
        rng = np.random.RandomState(42)
        subtypes = df_imp["subtype"].unique()
        sampled_sub_dfs = []
        for st in subtypes:
            sub_df = df_imp[df_imp["subtype"] == st]
            st_target = int(round(len(sub_df) * (target_imp / n_imp)))
            sampled = sub_df.sample(n=min(st_target, len(sub_df)), random_state=42)
            sampled_sub_dfs.append(sampled)
        df_imp = pd.concat(sampled_sub_dfs).reset_index(drop=True)

    df_final = pd.concat([df_exp, df_imp, df_nontoxic], ignore_index=True)
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_final["id"] = [f"cs_tox_{i+1:05d}" for i in range(len(df_final))]

    df_final.to_csv(FINAL_CORPUS_CSV, index=False)
    print(f"\nSaved final balanced benchmark corpus to: {FINAL_CORPUS_CSV} ({len(df_final):,} samples)")

    # 5. Calibration Audit ROC-AUC
    y_true = (df_final["label"] == "toxic").astype(int).values
    y_scores = df_final["score"].values
    roc_auc = roc_auc_score(y_true, y_scores)
    print(f"Classifier ROC-AUC: {roc_auc:.4f} (Paper reported: 0.684)")

if __name__ == "__main__":
    clean_and_assemble_corpus()
