import os
import json
from collections import defaultdict
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal
from statsmodels.stats.multitest import multipletests

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = "data/attributes/stabilityai-stable-diffusion-3"
FORMULATIONS = ["direct", "indirect"]
TARGET_KEYS = ["gender", "race", "age"]

# Name of the folder holding the neutral baseline counts
BASELINE_CONCEPT_NAME = "baseline"  # change to your folder name
# If you DON'T have a baseline folder, uncomment to hard-code gender baseline (counts out of 100)
# HARDCODE_BASELINE_GENDER = {"male": 89, "female": 11}
HARDCODE_BASELINE_GENDER = None

# Optional: If you want to fix the category order (recommended for consistent tables)
GENDER_ORDER = ["male", "female"]
RACE_ORDER   = ["caucasian", "afro-american", "latino-hispanic", "east asian",
                "southeast asian", "indian", "middle eastern"]
AGE_ORDER    = ["0-2", "3-9", "10-19", "20-29", "30-39",
                "40-49", "50-59", "60-69", "70+"]

# Where to save outputs
OUT_GENDER_CSV = "gender_tests.csv"
OUT_RACE_CSV   = "race_tests.csv"
OUT_AGE_CSV    = "age_tests.csv"

# -------------------------
# HELPERS
# -------------------------

def chi2_with_correction(table):
    """Run chi2, add 0.5 correction if expected freqs contain zeros."""
    try:
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        if (expected == 0).any():
            table = table + 0.5  # Haldane–Anscombe correction
            chi2, p, dof, expected = chi2_contingency(table, correction=False)
    except ValueError:
        # In case chi2 fails outright (rare), apply correction pre-emptively
        table = table + 0.5
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
    return chi2, p, table

def read_attributes(base_dir, formulations):
    """
    Returns:
      data[formulation][concept] -> dict with keys in TARGET_KEYS mapping to dicts of counts
    """
    data = {f: {} for f in formulations}
    # concept folders at BASE_DIR level
    concept_folders = sorted([f for f in os.listdir(base_dir)
                              if os.path.isdir(os.path.join(base_dir, f))])
    for concept in concept_folders:
        for form in formulations:
            json_path = os.path.join(base_dir, concept, form, "attributes_breakdown.json")
            if not os.path.exists(json_path):
                # It's OK for baseline concept to exist only under one formulation;
                # we'll handle baseline search below.
                continue
            with open(json_path, "r") as fh:
                jd = json.load(fh)
            data[form].setdefault(concept, {})
            for key in TARGET_KEYS:
                data[form][concept][key] = jd.get(key, {})
    return data
def get_baseline_counts(attribute, order=None):
    """
    Loads baseline counts for the given attribute from the control_images path.
    Returns dict label->count, and also the final order used.
    """
    baseline_path = os.path.join(
        "data", "control_images", "stabilityai-stable-diffusion-3", "attributes_breakdown.json"
    )

    if not os.path.exists(baseline_path):
        raise RuntimeError(f"Baseline file not found: {baseline_path}")

    with open(baseline_path, "r") as f:
        jd = json.load(f)

    baseline_counts = jd.get(attribute, None)

    if baseline_counts is None:
        raise RuntimeError(f"No baseline counts found for attribute '{attribute}'")

    # normalize to include all labels in order
    if order is None:
        order = sorted(baseline_counts.keys())

    norm = {lab: int(baseline_counts.get(lab, 0)) for lab in order}
    return norm, order


def counts_vector(counts_dict, order):
    """Return list of counts in fixed order."""
    return [int(counts_dict.get(k, 0)) for k in order]

def cramers_v_from_chi2(chi2, n, r, c):
    """Cramér's V effect size from chi-square for an r x c table."""
    if n == 0:
        return np.nan
    return np.sqrt(chi2 / (n * (min(r - 1, c - 1) if min(r - 1, c - 1) > 0 else 1)))

def epsilon_squared_kruskal(H, n_groups, N):
    """
    Epsilon-squared effect size for Kruskal–Wallis (semi-partial).
    H: statistic, n_groups: number of groups (here 2), N: total observations
    """
    if N <= 1:
        return np.nan
    return (H - (n_groups - 1)) / (N - 1)

def expand_age_counts_to_samples(counts_dict):
    """
    Convert age-bin counts to ordinal numeric samples (bin midpoints).
    This lets us run non-parametric tests. Midpoints are a reasonable ordinal proxy.
    """
    midpoints = {
        "0-2": 1, "3-9": 6, "10-19": 14.5, "20-29": 24.5, "30-39": 34.5,
        "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "70+": 75  # 75 as a rough open-ended midpoint
    }
    samples = []
    for k, v in counts_dict.items():
        mp = midpoints.get(k, None)
        if mp is None:
            continue
        samples.extend([mp] * int(v))
    return np.array(samples, dtype=float)

# -------------------------
# MAIN ANALYSIS
# -------------------------
def main():
    data = read_attributes(BASE_DIR, FORMULATIONS)

    # ----------------- GENDER -----------------
    baseline_gender, gender_order = get_baseline_counts("gender", order=GENDER_ORDER)
    baseline_vec_gender = counts_vector(baseline_gender, gender_order)

    gender_rows = []
    for form in FORMULATIONS:
        for concept, attrs in data[form].items():
            if concept == BASELINE_CONCEPT_NAME:
                continue
            counts = attrs.get("gender", {})
            concept_vec = counts_vector(counts, gender_order)
            table = np.array([baseline_vec_gender, concept_vec], dtype=float)

            chi2, p_val, table_corr = chi2_with_correction(table)
            n = table_corr.sum()
            V = cramers_v_from_chi2(chi2, n, 2, len(gender_order))

            # Direction wrt male share
            base_male_share = baseline_vec_gender[0] / max(sum(baseline_vec_gender), 1)
            concept_male_share = concept_vec[0] / max(sum(concept_vec), 1)
            direction = ("more_male" if concept_male_share > base_male_share
                        else "more_female" if concept_male_share < base_male_share
                        else "same")

            gender_rows.append({
                "concept": concept, "formulation": form,
                "test": "chi2", "stat": chi2, "p_raw": p_val,
                "cramers_v": V,
                "baseline_male": baseline_vec_gender[0], "baseline_female": baseline_vec_gender[1],
                "concept_male": concept_vec[0], "concept_female": concept_vec[1],
                "direction_vs_baseline": direction
            })

    gender_df = pd.DataFrame(gender_rows)

    # FDR per formulation
    gender_df["p_adj"] = np.nan
    gender_df["significant_fdr_0.05"] = False
    for form in FORMULATIONS:
        mask = gender_df["formulation"] == form
        if mask.any():
            _, p_adj, _, _ = multipletests(gender_df.loc[mask, "p_raw"], alpha=0.05, method="fdr_bh")
            gender_df.loc[mask, "p_adj"] = p_adj
            gender_df.loc[mask, "significant_fdr_0.05"] = p_adj < 0.05

    gender_df.sort_values(["formulation", "p_adj"], inplace=True)
    gender_df.to_csv(OUT_GENDER_CSV, index=False)

    # ----------------- RACE -----------------
    baseline_race, race_order = get_baseline_counts("race", order=RACE_ORDER)
    baseline_vec_race = counts_vector(baseline_race, race_order)

    race_rows = []
    for form in FORMULATIONS:
        for concept, attrs in data[form].items():
            if concept == BASELINE_CONCEPT_NAME:
                continue
            counts = attrs.get("race", {})
            concept_vec = counts_vector(counts, race_order)

            table = np.array([baseline_vec_race, concept_vec], dtype=float)
            chi2, p_val, table_corr = chi2_with_correction(table)
            n = table_corr.sum()
            V = cramers_v_from_chi2(chi2, n, 2, len(race_order))

            race_rows.append({
                "concept": concept, "formulation": form,
                "test": "chi2", "stat": chi2, "p_raw": p_val,
                "cramers_v": V,
                **{f"baseline_{lab}": baseline_vec_race[i] for i, lab in enumerate(race_order)},
                **{f"concept_{lab}": concept_vec[i] for i, lab in enumerate(race_order)},
            })
    race_df = pd.DataFrame(race_rows)
    race_df["p_adj"] = np.nan
    race_df["significant_fdr_0.05"] = False
    for form in FORMULATIONS:
        mask = race_df["formulation"] == form
        if mask.any():
            _, p_adj, _, _ = multipletests(race_df.loc[mask, "p_raw"], alpha=0.05, method="fdr_bh")
            race_df.loc[mask, "p_adj"] = p_adj
            race_df.loc[mask, "significant_fdr_0.05"] = p_adj < 0.05

    race_df.sort_values(["formulation", "p_adj"], inplace=True)
    race_df.to_csv(OUT_RACE_CSV, index=False)

    # ----------------- AGE -----------------
    # Age test: Kruskal–Wallis on ordinal midpoints, expanding counts into samples
    baseline_age, age_order = get_baseline_counts("age", order=AGE_ORDER)
    base_samples = expand_age_counts_to_samples(baseline_age)

    age_rows = []
    for form in FORMULATIONS:
        for concept, attrs in data[form].items():
            if concept == BASELINE_CONCEPT_NAME:
                continue
            counts = attrs.get("age", {})
            # Expand counts to ordinal sample values
            concept_samples = expand_age_counts_to_samples(counts)

            # Only test if both have at least 2 observations
            if len(base_samples) < 2 or len(concept_samples) < 2:
                H, p_val, eps2 = np.nan, np.nan, np.nan
            else:
                H, p_val = kruskal(base_samples, concept_samples, nan_policy="omit")
                N = len(base_samples) + len(concept_samples)
                eps2 = epsilon_squared_kruskal(H, n_groups=2, N=N)

            # Also keep raw counts (helps readers)
            concept_vec = counts_vector(counts, age_order)
            age_rows.append({
                "concept": concept, "formulation": form,
                "test": "kruskal", "stat": H, "p_raw": p_val,
                "epsilon_sq": eps2,
                **{f"baseline_{lab}": baseline_age.get(lab, 0) for lab in age_order},
                **{f"concept_{lab}": concept_vec[i] for i, lab in enumerate(age_order)},
            })

    age_df = pd.DataFrame(age_rows)
    age_df["p_adj"] = np.nan
    age_df["significant_fdr_0.05"] = False
    for form in FORMULATIONS:
        mask = age_df["formulation"] == form
        if mask.any():
            # Drop NaNs for correction, then write back
            idx = age_df.loc[mask, "p_raw"].dropna().index
            if len(idx) > 0:
                _, p_adj, _, _ = multipletests(age_df.loc[idx, "p_raw"], alpha=0.05, method="fdr_bh")
                age_df.loc[idx, "p_adj"] = p_adj
                age_df.loc[idx, "significant_fdr_0.05"] = p_adj < 0.05

    age_df.sort_values(["formulation", "p_adj"], inplace=True)
    age_df.to_csv(OUT_AGE_CSV, index=False)

    # ----------------- SUMMARIES -----------------
    def summarize(df, label):
        summ = []
        for form in FORMULATIONS:
            sub = df[df["formulation"] == form]
            n_total = sub.shape[0]
            n_sig = int(sub["significant_fdr_0.05"].sum())
            summ.append((form, n_sig, n_total))
        print(f"\n[{label}] Significant (FDR<.05) per formulation:")
        for form, n_sig, n_total in summ:
            print(f"  {form:>8}: {n_sig}/{n_total}")

    summarize(gender_df, "Gender")
    summarize(race_df,   "Race")
    summarize(age_df,    "Age")

    # Extra: quick direction counts for gender
    for form in FORMULATIONS:
        sub = gender_df[(gender_df["formulation"] == form) & (gender_df["significant_fdr_0.05"])]
        more_male = (sub["direction_vs_baseline"] == "more_male").sum()
        more_female = (sub["direction_vs_baseline"] == "more_female").sum()
        print(f"\n[Gender] Direction among significant ({form}): "
              f"{more_male} more_male vs {more_female} more_female")

    print("\nCSV files written:")
    print(f"  - {OUT_GENDER_CSV}")
    print(f"  - {OUT_RACE_CSV}")
    print(f"  - {OUT_AGE_CSV}")

if __name__ == "__main__":
    main()
