import math
import os
import random
import pandas as pd
from config import PILOT_RESULTS_CSV, CONFIDENCE_LEVEL, SUBTYPE_PRIOR_WEIGHT, SUBTYPE_TO_LABEL, PILOT_ALPHA_SMOOTHING

JEFFREYS_ALPHA = PILOT_ALPHA_SMOOTHING
BANDS = ["easy", "medium", "hard"]
NUM_BANDS = len(BANDS)

def load_pilot_matrix() -> pd.DataFrame:
    if not os.path.exists(PILOT_RESULTS_CSV):
        raise FileNotFoundError(
            f"Pilot file not found: {PILOT_RESULTS_CSV}. Run step1_pilot_estimation.py first!"
        )
    df = pd.read_csv(PILOT_RESULTS_CSV)
    return df

def apply_zero_count_smoothing(row: pd.Series) -> dict:
    """
    Implements Eq. (1) in Paper:
    P_tilde(b | t) = (c(t,b) + alpha) / (N_t + 3 * alpha) if any target band is empty,
    else c(t,b) / N_t.
    """
    total = row["total_samples"]
    counts = {b: int(row[f"count_{b}"]) for b in BANDS}
    has_empty_target_band = any(c == 0 for c in counts.values())
    if not has_empty_target_band:
        return {b: float(row[f"p_{b}"]) for b in BANDS}
    return {
        b: (counts[b] + JEFFREYS_ALPHA) / (total + JEFFREYS_ALPHA * NUM_BANDS)
        for b in BANDS
    }

def load_smoothed_matrix() -> pd.DataFrame:
    df = load_pilot_matrix()
    for i, row in df.iterrows():
        smoothed_p = apply_zero_count_smoothing(row)
        for b in BANDS:
            df.at[i, f"p_{b}"] = smoothed_p[b]
    return df

def get_sampling_weights_for_band(band: str, pilot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements Eq. (2) in Paper:
    P_select(t | b) = P_tilde(b | t) * w_prior(t) / sum_u(P_tilde(b | u) * w_prior(u))
    """
    col = f"p_{band}"
    df = pilot_df.copy()
    df["_prior"] = df["subtype"].map(SUBTYPE_PRIOR_WEIGHT).fillna(1e-6)
    df["_w"] = df[col].clip(lower=1e-6) * df["_prior"]
    df["_w"] = df["_w"] / df["_w"].sum()
    return df

def pick_subtype_weighted(band: str, pilot_df: pd.DataFrame = None, rng: random.Random = None, exclude: set = None):
    """Selects a subtype for the target band using Eq. (2)."""
    if pilot_df is None:
        pilot_df = load_smoothed_matrix()
    if rng is None:
        rng = random

    col = f"p_{band}"
    sub_df = get_sampling_weights_for_band(band, pilot_df)

    if exclude:
        sub_df = sub_df[~sub_df["subtype"].isin(exclude)]
        if sub_df.empty:
            return None, None, 0.0
        sub_df = sub_df.copy()
        sub_df["_w"] = sub_df["_w"] / sub_df["_w"].sum()

    subtype = rng.choices(sub_df["subtype"].tolist(), weights=sub_df["_w"].tolist(), k=1)[0]
    p_hat = float(sub_df.loc[sub_df["subtype"] == subtype, col].iloc[0])
    return subtype, SUBTYPE_TO_LABEL[subtype], p_hat