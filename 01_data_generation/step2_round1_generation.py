"""
step2_round1_generation.py
Step 2: Round 1 Data Generation using Claude 3 Haiku via OpenRouter.
Aligns with Paper Section 3.1.2 & 3.3:
- Targets 7,755 samples using weighted subtype sampling (Eq. 2).
- Filters candidates via threshold tau_gen = 0.3.
- Writes to data/round1_7755.csv.
"""
import os
import sys
import time
import argparse
import random
import re
import pandas as pd
from openai import OpenAI

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, ROUND1_MODEL, ROUND1_TEMPERATURE,
    GENERATION_THRESHOLD, BAND_TARGET_PROPORTIONS, ROUND1_RESULTS_CSV
)
from scenarios import COMPLAINT_SCENARIOS, IDENTITY_BIAS_SCENARIOS, INDUSTRIES, REGISTERS, LENGTH_BUCKETS
from prompts import SYSTEM_PROMPT, build_batch_user_prompt
from scorer import score_batch, get_band
from weighted_sampler import pick_subtype_weighted

TARGET_TOTAL = 7755
BATCH_ITEMS = 5

def parse_msg_tags(raw_text: str, num_items: int) -> list:
    results = []
    for i in range(1, num_items + 1):
        pattern = rf"<msg_{i}>(.*?)</msg_{i}>"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        results.append(match.group(1).strip().strip('"').strip("'") if match else "")
    return results

def run_round1_generation(force_restart: bool = False):
    print("=" * 80)
    print(f"ROUND 1: CLAUDE 3 HAIKU GENERATION (Target: {TARGET_TOTAL:,} messages)")
    print(f"Model: {ROUND1_MODEL} | Temperature: {ROUND1_TEMPERATURE} | API: OpenRouter")
    print("=" * 80)
    
    if os.path.exists(ROUND1_RESULTS_CSV) and not force_restart:
        df_existing = pd.read_csv(ROUND1_RESULTS_CSV)
        print(f"Found existing Round 1 dataset: {ROUND1_RESULTS_CSV} ({len(df_existing):,} samples).")
        print("Pass --force to restart generation from scratch.")
        return

    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not configured. Please set OPENROUTER_API_KEY in .env.")
        sys.exit(1)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    
    # Initialize or resume
    if os.path.exists(ROUND1_RESULTS_CSV) and force_restart:
        os.remove(ROUND1_RESULTS_CSV)
        
    collected_rows = []
    current_count = 0
    rng = random.Random(42)
    
    band_targets = {
        "easy": int(TARGET_TOTAL * BAND_TARGET_PROPORTIONS["easy"]),
        "medium": int(TARGET_TOTAL * BAND_TARGET_PROPORTIONS["medium"]),
        "hard": int(TARGET_TOTAL * BAND_TARGET_PROPORTIONS["hard"])
    }
    band_counts = {"easy": 0, "medium": 0, "hard": 0}
    
    print(f"Target band distribution: {band_targets}")
    print("Starting generation loop...")
    
    while current_count < TARGET_TOTAL:
        # Determine target band needing samples
        underfilled = [b for b, tgt in band_targets.items() if band_counts[b] < tgt]
        target_b = rng.choice(underfilled) if underfilled else "easy"
        
        subtasks = []
        for _ in range(BATCH_ITEMS):
            subtype, label, _ = pick_subtype_weighted(target_b, rng)
            ind = rng.choice(INDUSTRIES)
            reg = rng.choice(REGISTERS)
            lmin, lmax = rng.choice(LENGTH_BUCKETS)
            
            is_bias = (rng.random() < 0.2)
            scen = rng.choice(IDENTITY_BIAS_SCENARIOS) if is_bias else rng.choice(COMPLAINT_SCENARIOS)
            subtasks.append({
                "label": label,
                "subtype": subtype,
                "industry": ind,
                "register": reg,
                "scenario": scen,
                "topic_type": "identity_bias" if is_bias else "complaint",
                "min_words": lmin,
                "max_words": lmax,
                "target_band": target_b
            })
            
        user_prompt = build_batch_user_prompt(subtasks)
        try:
            resp = client.chat.completions.create(
                model=ROUND1_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=ROUND1_TEMPERATURE,
                max_tokens=650
            )
            raw = resp.choices[0].message.content
            texts = parse_msg_tags(raw, BATCH_ITEMS)
            scores = score_batch(texts)
            
            for task, text, score in zip(subtasks, texts, scores):
                if not text:
                    continue
                band = get_band(score)
                # Keep if valid passing or non-toxic
                if score >= GENERATION_THRESHOLD or task["label"] == "nontoxic":
                    row = {
                        "id": f"r1_{current_count + 1:05d}",
                        "text": text,
                        "label": task["label"],
                        "subtype": task["subtype"],
                        "topic_type": task["topic_type"],
                        "scenario": task["scenario"],
                        "industry": task["industry"],
                        "register": task["register"],
                        "score": round(score, 6),
                        "band": band
                    }
                    collected_rows.append(row)
                    current_count += 1
                    if band in band_counts:
                        band_counts[band] += 1
                        
            print(f"Generated: {current_count:,}/{TARGET_TOTAL:,} samples (Easy: {band_counts['easy']}, Med: {band_counts['medium']}, Hard: {band_counts['hard']})", end="\r")
            
            # Periodic flush every 50 samples
            if len(collected_rows) >= 50:
                df_chunk = pd.DataFrame(collected_rows)
                header = not os.path.exists(ROUND1_RESULTS_CSV)
                df_chunk.to_csv(ROUND1_RESULTS_CSV, mode="a", header=header, index=False)
                collected_rows = []
                
        except Exception as e:
            print(f"\nAPI Error: {e}. Backing off for 3 seconds...")
            time.sleep(3)

    if collected_rows:
        df_chunk = pd.DataFrame(collected_rows)
        header = not os.path.exists(ROUND1_RESULTS_CSV)
        df_chunk.to_csv(ROUND1_RESULTS_CSV, mode="a", header=header, index=False)

    print(f"\nCompleted Round 1 generation! Saved to: {ROUND1_RESULTS_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Round 1 Data Generation")
    parser.add_argument("--force", action="store_true", help="Force regenerate from scratch")
    args = parser.parse_args()
    
    run_round1_generation(force_restart=args.force)
