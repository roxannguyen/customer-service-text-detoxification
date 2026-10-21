"""
config.py
Central Configuration for Dataset Generation Pipeline
Strictly aligned with Paper Section 3.1 and 3.2.
"""
import os
import math
from pathlib import Path
from scipy.stats import norm
from dotenv import load_dotenv

load_dotenv()

# --- Directory Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PILOT_RESULTS_CSV = DATA_DIR / "pilot_results.csv"
ROUND1_RESULTS_CSV = DATA_DIR / "round1_7755.csv"
FINAL_CORPUS_CSV = DATA_DIR / "final_corpus_9631.csv"

# --- Model Configurations ---
# Round 1: Claude 3 Haiku via OpenRouter API (Paper Section 3.1.2)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
ROUND1_MODEL = "anthropic/claude-3-haiku"
ROUND1_TEMPERATURE = 0.96

# Supplementary Round 2: Qwen 27B via GroqCloud API (Paper Section 3.1.2)
raw_groq = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
GROQ_API_KEYS = [k.strip() for k in raw_groq.split(",") if k.strip()]
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
ROUND2_MODEL = "qwen/qwen3.8-27b"
ROUND2_TEMPERATURE = 0.95

# --- Scorer Configuration (Paper Section 3.1.3) ---
SCORER_MODEL = "unitary/unbiased-toxic-roberta"
GENERATION_THRESHOLD = 0.3  # tau_gen = 0.3
EASY_MAX = 0.6              # Easy: [0.3, 0.6)
MEDIUM_MAX = 0.8            # Medium: [0.6, 0.8), Hard: >= 0.8

# Target Band Distribution matching Jigsaw / TRIAD: 47.4% Easy, 31.1% Medium, 21.5% Hard
BAND_TARGET_PROPORTIONS = {
    "easy": 0.474,
    "medium": 0.311,
    "hard": 0.215
}

# --- Pilot Estimation Parameters (Paper Section 3.2.1) ---
CONFIDENCE_LEVEL = 0.90
MARGIN_OF_ERROR = 0.10
PILOT_ALPHA_SMOOTHING = 0.5  # alpha = 0.5 in Eq. (1)

def calculate_pilot_sample_size(confidence: float = CONFIDENCE_LEVEL,
                                margin_of_error: float = MARGIN_OF_ERROR,
                                p: float = 0.5) -> int:
    z = norm.ppf(1 - (1 - confidence) / 2)  # z = 1.645 for 90% confidence
    n = (z ** 2) * p * (1 - p) / (margin_of_error ** 2)
    return math.ceil(n)  # returns 68

PILOT_SAMPLES_PER_SUBTYPE = calculate_pilot_sample_size()  # n = 68

# --- Subtype Prior Weights w_prior(t) for Eq. (2) ---
SUBTYPE_PRIOR_WEIGHT = {
    "explicit": 0.0200,
    "explicit_mild_profanity": 0.0100,
    "contempt": 0.0400,
    "sarcasm": 0.2000,
    "belittling_competence": 0.1800,
    "implicit_threat": 0.2000,
    "passive_aggressive": 0.1800,
    "neutral_polite": 0.0500,
    "harsh_but_civil_criticism": 0.1100,
    "legitimate_frustration_at_product": 0.0100,
}

SUBTYPE_TO_LABEL = {
    "explicit": "toxic",
    "explicit_mild_profanity": "toxic",
    "contempt": "toxic",
    "sarcasm": "toxic",
    "belittling_competence": "toxic",
    "implicit_threat": "toxic",
    "passive_aggressive": "toxic",
    "neutral_polite": "nontoxic",
    "harsh_but_civil_criticism": "nontoxic",
    "legitimate_frustration_at_product": "nontoxic",
}

MACRO_LABEL_MAP = {
    "explicit": "Toxic Explicit",
    "explicit_mild_profanity": "Toxic Explicit",
    "sarcasm": "Toxic Implicit",
    "belittling_competence": "Toxic Implicit",
    "passive_aggressive": "Toxic Implicit",
    "implicit_threat": "Toxic Implicit",
    "contempt": "Toxic Implicit",
    "legitimate_frustration_at_product": "Non-toxic Frustration",
    "neutral_polite": "Non-toxic Civil",
    "harsh_but_civil_criticism": "Non-toxic Civil"
}
