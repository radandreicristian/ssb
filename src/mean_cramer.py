import pandas as pd

# Load your CSV
df = pd.read_csv("race_tests.csv")

# Only keep rows that were significant after FDR
sig_df = df[df["significant_fdr_0.05"] == True]

# Group by formulation and compute stats
summary = (
    sig_df.groupby("formulation")["cramers_v"]
    .agg(["count", "mean", "median", "min", "max"])
)

print(summary)

import pandas as pd

# Load your CSV
df = pd.read_csv("age_tests.csv")

# Only keep rows that were significant after FDR
sig_df = df[df["significant_fdr_0.05"] == True]

# Group by formulation and compute descriptive stats for epsilon²
summary = (
    sig_df.groupby("formulation")["epsilon_sq"]
    .agg(["count", "mean", "median", "min", "max"])
)

print(summary)