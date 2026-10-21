"""
step3_round2_supplementary.py
Step 3: Supplementary Round Data Generation using Qwen via GroqCloud API.
Aligns with Paper Section 3.1.2 & 3.3:
- Targets 2,533 candidate messages of Toxic Explicit to balance the corpus.
- Employs 3-word opening prefix diversity directives to prevent formulaic outputs.
- Saves candidate pool to data/raw_round2_candidates_2533.csv.
"""
import os
import sys
import time
import argparse
import random
import re
from collections import defaultdict
import pandas as pd
from openai import OpenAI

from config import (
    GROQ_API_KEYS, GROQ_BASE_URL, ROUND2_MODEL, ROUND2_TEMPERATURE,
    DATA_DIR, GENERATION_THRESHOLD
)
from scenarios import COMPLAINT_SCENARIOS, INDUSTRIES, REGISTERS, LENGTH_BUCKETS
from prompts import SYSTEM_PROMPT
from scorer import score_batch, get_band

RAW_ROUND2_CSV = DATA_DIR / "raw_round2_candidates_2533.csv"
TARGET_CANDIDATES = 2533
BATCH_SIZE = 5

OPENING_PREFIX_MAX = 3

EXPLICIT_PROMPT_TEMPLATE = """Generate {n} realistic, angry customer complaint messages targeting customer support.
Specifications:
- Label: TOXIC
- Subtype: explicit (direct insults, name-calling, profanity directed at staff or company)
- Industry: {industry}
- Channel: {register}
- Scenario: {scenario}
- Length: approx {min_words}-{max_words} words.

CRITICAL INSTRUCTIONS:
1. Do NOT repeat formulaic opening words. Start messages with diverse words and styles.
2. Output each message wrapped in <msg_1> to <msg_{n}> tags.
3. Output ONLY the customer messages with no introduction, numbering, or explanations.
"""

def parse_msg_tags(raw_text: str, num_items: int) -> list:
    results = []
    for i in range(1, num_items + 1):
        pattern = rf"<msg_{i}>(.*?)</msg_{i}>"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        results.append(match.group(1).strip().strip('"').strip("'") if match else "")
    return results

def get_prefix_3words(text: str) -> str:
    words = re.findall(r"\b\w+\b", text.lower())
    return " ".join(words[:3]) if len(words) >= 3 else " ".join(words)

def run_supplementary_generation(force_restart: bool = False):
    print("=" * 80)
    print(f"SUPPLEMENTARY ROUND: QWEN GENERATION (Target: {TARGET_CANDIDATES:,} explicit candidates)")
    print(f"Model: {ROUND2_MODEL} | Temperature: {ROUND2_TEMPERATURE} | API: GroqCloud")
    print("=" * 80)

    if os.path.exists(RAW_ROUND2_CSV) and not force_restart:
        df_raw = pd.read_csv(RAW_ROUND2_CSV)
        print(f"Found existing Round 2 candidates: {RAW_ROUND2_CSV} ({len(df_raw):,} samples).")
        print("Pass --force to restart generation from scratch.")
        return

    if not GROQ_API_KEYS:
        print("Error: GROQ_API_KEYS not configured. Please set GROQ_API_KEYS in .env.")
        sys.exit(1)

    clients = [OpenAI(api_key=key, base_url=GROQ_BASE_URL) for key in GROQ_API_KEYS]
    client_idx = 0

    if os.path.exists(RAW_ROUND2_CSV) and force_restart:
        os.remove(RAW_ROUND2_CSV)

    prefix_counts = defaultdict(int)
    collected = []
    current_count = 0
    rng = random.Random(42)

    print("Starting supplementary generation loop...")

    while current_count < TARGET_CANDIDATES:
        client = clients[client_idx % len(clients)]
        client_idx += 1

        ind = rng.choice(INDUSTRIES)
        reg = rng.choice(REGISTERS)
        scen = rng.choice(COMPLAINT_SCENARIOS)
        lmin, lmax = rng.choice(LENGTH_BUCKETS)

        prompt = EXPLICIT_PROMPT_TEMPLATE.format(
            n=BATCH_SIZE,
            industry=ind,
            register=reg,
            scenario=scen,
            min_words=lmin,
            max_words=lmax
        )

        try:
            resp = client.chat.completions.create(
                model=ROUND2_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=ROUND2_TEMPERATURE,
                max_tokens=600
            )
            raw = resp.choices[0].message.content
            texts = parse_msg_tags(raw, BATCH_SIZE)
            scores = score_batch(texts)

            for txt, sc in zip(texts, scores):
                if not txt:
                    continue
                # Diversity filter on 3-word opening prefix
                pfx = get_prefix_3words(txt)
                if prefix_counts[pfx] >= OPENING_PREFIX_MAX:
                    continue
                prefix_counts[pfx] += 1

                row = {
                    "id": f"r2_{current_count + 1:05d}",
                    "text": txt,
                    "label": "toxic",
                    "subtype": "explicit",
                    "topic_type": "complaint",
                    "scenario": scen,
                    "industry": ind,
                    "register": reg,
                    "score": round(sc, 6),
                    "band": get_band(sc)
                }
                collected.append(row)
                current_count += 1

            print(f"Generated Explicit candidates: {current_count:,}/{TARGET_CANDIDATES:,}", end="\r")

            if len(collected) >= 50:
                df_chunk = pd.DataFrame(collected)
                header = not os.path.exists(RAW_ROUND2_CSV)
                df_chunk.to_csv(RAW_ROUND2_CSV, mode="a", header=header, index=False)
                collected = []

        except Exception as e:
            print(f"\nAPI Error: {e}. Retrying in 2 seconds...")
            time.sleep(2)

    if collected:
        df_chunk = pd.DataFrame(collected)
        header = not os.path.exists(RAW_ROUND2_CSV)
        df_chunk.to_csv(RAW_ROUND2_CSV, mode="a", header=header, index=False)

    print(f"\nCompleted Supplementary Generation! Saved to: {RAW_ROUND2_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Supplementary Explicit Generation")
    parser.add_argument("--force", action="store_true", help="Force regenerate from scratch")
    args = parser.parse_args()

    run_supplementary_generation(force_restart=args.force)
