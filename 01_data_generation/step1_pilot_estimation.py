"""
step1_pilot_estimation.py
Step 1: Pilot Estimation and Verification of Paper Table 1.
Aligns with Paper Section 3.2.1:
- Generates or loads empirical pilot samples (n = 68 per subtype across 10 subtypes, total 680).
- Computes transition matrix P_hat(band | subtype) across difficulty bands.
- Validates zero-count smoothing with alpha = 0.5 (Eq. 1).
"""
import os
import sys
import argparse
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI

from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, ROUND1_MODEL, ROUND1_TEMPERATURE,
    PILOT_RESULTS_CSV, PILOT_SAMPLES_PER_SUBTYPE, GENERATION_THRESHOLD,
    EASY_MAX, MEDIUM_MAX, SUBTYPE_TO_LABEL
)
from scenarios import COMPLAINT_SCENARIOS, IDENTITY_BIAS_SCENARIOS, INDUSTRIES, REGISTERS, LENGTH_BUCKETS
from prompts import SYSTEM_PROMPT, build_batch_user_prompt
from scorer import score_batch, get_band
from weighted_sampler import load_smoothed_matrix

def parse_msg_tags(raw_text: str, num_items: int) -> list:
    results = []
    for i in range(1, num_items + 1):
        pattern = rf"<msg_{i}>(.*?)</msg_{i}>"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        results.append(match.group(1).strip().strip('"').strip("'") if match else "")
    return results

def run_pilot_generation():
    """Generates n=68 samples for each of the 10 subtypes using Claude 3 Haiku."""
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not configured. Please set OPENROUTER_API_KEY in .env.")
        sys.exit(1)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    subtypes = list(SUBTYPE_TO_LABEL.keys())
    batch_size = 5
    
    print("=" * 80)
    print(f"RUNNING PILOT GENERATION: {PILOT_SAMPLES_PER_SUBTYPE} samples x {len(subtypes)} subtypes = {PILOT_SAMPLES_PER_SUBTYPE * len(subtypes)} total")
    print(f"Model: {ROUND1_MODEL} | API: OpenRouter")
    print("=" * 80)
    
    pilot_rows = []
    
    for st in subtypes:
        label = SUBTYPE_TO_LABEL[st]
        collected = []
        print(f"\nGenerating subtype: '{st}' ({label})...")
        
        while len(collected) < PILOT_SAMPLES_PER_SUBTYPE:
            n_need = min(batch_size, PILOT_SAMPLES_PER_SUBTYPE - len(collected))
            subtasks = []
            for _ in range(n_need):
                is_id_bias = (len(collected) % 5 == 0)
                subtasks.append({
                    "label": label,
                    "subtype": st,
                    "industry": INDUSTRIES[len(collected) % len(INDUSTRIES)],
                    "register": REGISTERS[len(collected) % len(REGISTERS)],
                    "scenario": (IDENTITY_BIAS_SCENARIOS[len(collected) % len(IDENTITY_BIAS_SCENARIOS)]
                                 if is_id_bias else COMPLAINT_SCENARIOS[len(collected) % len(COMPLAINT_SCENARIOS)]),
                    "topic_type": "identity_bias" if is_id_bias else "complaint",
                    "min_words": 15,
                    "max_words": 40
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
                    max_tokens=600
                )
                texts = parse_msg_tags(resp.choices[0].message.content, n_need)
                scores = score_batch(texts)
                for txt, sc in zip(texts, scores):
                    if txt:
                        collected.append({"subtype": st, "text": txt, "score": sc, "band": get_band(sc)})
                print(f"  Collected: {len(collected)}/{PILOT_SAMPLES_PER_SUBTYPE}", end="\r")
            except Exception as e:
                print(f"\n  API error: {e}. Retrying in 2s...")
                time.sleep(2)
        
        # Calculate empirical transition probabilities for this subtype
        df_st = pd.DataFrame(collected)
        p_below = (df_st["score"] < GENERATION_THRESHOLD).mean()
        p_easy = (df_st["band"] == "easy").mean()
        p_med = (df_st["band"] == "medium").mean()
        p_hard = (df_st["band"] == "hard").mean()
        
        pilot_rows.append({
            "subtype": st,
            "samples": len(df_st),
            "p_below_threshold": p_below,
            "p_easy": p_easy,
            "p_medium": p_med,
            "p_hard": p_hard
        })

    df_pilot = pd.DataFrame(pilot_rows)
    df_pilot.to_csv(PILOT_RESULTS_CSV, index=False)
    print(f"\nSaved pilot estimation results to: {PILOT_RESULTS_CSV}")

def display_table_1():
    print("=" * 80)
    print(f"TABLE 1: Pilot results (n={PILOT_SAMPLES_PER_SUBTYPE} per subtype, tau_gen={GENERATION_THRESHOLD})")
    print("Share of messages (%) below tau_gen and in each band.")
    print("=" * 80)
    
    if not os.path.exists(PILOT_RESULTS_CSV):
        print(f"Notice: {PILOT_RESULTS_CSV} not found. Run with --generate to generate via API.")
        return

    df = pd.read_csv(PILOT_RESULTS_CSV)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Subtype": r["subtype"].replace("_", " "),
            "Below (%)": f"{r['p_below_threshold'] * 100:.1f}",
            "Easy (%)": f"{r['p_easy'] * 100:.1f}",
            "Med. (%)": f"{r['p_medium'] * 100:.1f}",
            "Hard (%)": f"{r['p_hard'] * 100:.1f}"
        })
    df_tab1 = pd.DataFrame(rows)
    print(df_tab1.to_string(index=False))
    print("=" * 80)
    print("Zero-count Smoothing Validation (Eq. 1):")
    load_smoothed_matrix()
    print("Successfully validated smoothed transition matrix with alpha=0.5.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Pilot Estimation & Table 1 Verification")
    parser.add_argument("--generate", action="store_true", help="Execute live generation via API")
    args = parser.parse_args()
    
    if args.generate:
        run_pilot_generation()
    display_table_1()
