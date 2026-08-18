"""
Inferential statistics on the movie dataset: confidence intervals,
hypothesis testing (ANOVA + pairwise comparisons), and multilinear
regression, used together to answer "does genre actually affect box office
performance, or does it just look that way in the sample."

Three techniques, three different questions:
  1. Confidence intervals on mean revenue by genre -- how much uncertainty
     is there around each genre's estimated average, given sample size.
  2. One-way ANOVA -- across all 8 genres at once, is there a statistically
     significant difference in mean revenue anywhere, before drilling into
     any specific pairwise comparison (running 8 separate t-tests without
     this omnibus test first inflates the false-positive rate).
  3. Multilinear regression -- how much of revenue is explained by budget,
     critic score, and runtime simultaneously, holding genre fixed, which
     answers a different question than any single-variable comparison can.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data():
    return pd.read_csv(DATA_DIR / "movies.csv")


def confidence_intervals_by_genre(df: pd.DataFrame, confidence=0.95):
    print(f"=== {confidence:.0%} Confidence Intervals: mean revenue by genre ===")
    results = []
    for genre, group in df.groupby("genre"):
        n = len(group)
        mean = group["revenue_millions"].mean()
        sem = stats.sem(group["revenue_millions"])
        margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
        results.append((genre, mean, mean - margin, mean + margin, n))
        print(f"  {genre:10s}: ${mean:6.1f}M  (95% CI: ${mean - margin:6.1f}M to ${mean + margin:6.1f}M, n={n})")
    return pd.DataFrame(results, columns=["genre", "mean_revenue", "ci_low", "ci_high", "n"])


def one_way_anova(df: pd.DataFrame):
    print("\n=== One-way ANOVA: does revenue differ significantly by genre? ===")
    groups = [group["revenue_millions"].values for _, group in df.groupby("genre")]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"  F-statistic = {f_stat:.2f}, p-value = {p_value:.6f}")
    if p_value < 0.05:
        print("  Reject null hypothesis: at least one genre's mean revenue differs significantly.")
    else:
        print("  Fail to reject null hypothesis: no significant difference detected across genres.")
    return f_stat, p_value


def pairwise_comparison(df: pd.DataFrame, genre_a="Horror", genre_b="Drama"):
    print(f"\n=== Pairwise t-test: {genre_a} vs. {genre_b} revenue ===")
    a = df[df["genre"] == genre_a]["revenue_millions"]
    b = df[df["genre"] == genre_b]["revenue_millions"]
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    print(f"  {genre_a} mean: ${a.mean():.1f}M | {genre_b} mean: ${b.mean():.1f}M")
    print(f"  t = {t_stat:.2f}, p-value = {p_value:.6f}")
    print("  Statistically significant difference" if p_value < 0.05 else "  Not statistically significant")
    return t_stat, p_value


def multilinear_regression(df: pd.DataFrame):
    print("\n=== Multilinear Regression: revenue ~ budget + critic_score + runtime + genre ===")
    model = smf.ols(
        "revenue_millions ~ budget_millions + critic_score + runtime_minutes + C(genre)",
        data=df
    ).fit()
    print(model.summary())

    print(f"\n  R-squared: {model.rsquared:.3f} "
          f"({model.rsquared:.1%} of revenue variance explained by this model)")
    print(f"  Budget coefficient: for every additional $1M in budget, "
          f"revenue increases by ${model.params['budget_millions']:.2f}M on average, "
          f"holding critic score, runtime, and genre fixed "
          f"(p = {model.pvalues['budget_millions']:.6f})")

    return model


def run():
    df = load_data()
    print(f"Loaded {len(df)} movies across {df['genre'].nunique()} genres.\n")

    confidence_intervals_by_genre(df)
    one_way_anova(df)
    pairwise_comparison(df, "Horror", "Drama")
    multilinear_regression(df)


if __name__ == "__main__":
    run()
