# BoxOfficeSignal

Inferential statistics on a movie dataset: confidence intervals, ANOVA and
pairwise hypothesis testing, and multilinear regression, used together to
answer "does genre actually affect box office performance, or does the
observed difference just look real in this sample."

## Why one technique isn't enough

A confidence interval on Horror's mean revenue vs. Drama's mean revenue can
look like a real difference just from eyeballing two ranges that don't
overlap, but "looks different" and "is statistically significant" aren't
the same claim, so a formal hypothesis test backs the eyeball comparison.
And neither a confidence interval nor a two-group t-test can answer "how
much of revenue is explained by budget and quality, independent of genre,"
which is what the multilinear regression is for. Each technique here
answers a specific, different question; none of them substitutes for
another.

## Repo layout

```
data_generation.py             synthetic movie dataset with genre-specific revenue patterns
scripts/inferential_stats.py   confidence intervals, ANOVA, pairwise t-test, multilinear regression
data/                           generated CSV (not committed)
```

## Why the dataset is synthetic

Real box-office datasets (Box Office Mojo, The Numbers, various Kaggle
mirrors) exist but carry inconsistent scraping/redistribution terms. This
dataset is generated with a documented, genre-specific revenue-to-budget
relationship (Horror's famous high ROI on small budgets, Drama's more
modest multiples, a mild critic-score and summer-release bump) so every
statistical result below can be checked against the known ground truth
that generated it.

## Key decisions

- **ANOVA runs before any pairwise comparison, not instead of it.** Testing all 8 genres against each other individually (28 pairwise t-tests) inflates the false-positive rate; a single omnibus ANOVA answers "is there a significant difference anywhere" first, and a specific pairwise test (Horror vs. Drama, in `pairwise_comparison()`) is only run second, as a follow-up on a specific question, not a fishing expedition across every possible pair.
- **Confidence intervals use the t-distribution, not the normal distribution.** Since the population standard deviation isn't known and is being estimated from each genre's sample (as small as n=63 for Thriller), the t-distribution is the statistically correct choice; using the normal distribution here would understate the true uncertainty, especially for the smaller genre samples.
- **The regression includes genre as a categorical fixed effect alongside budget, critic score, and runtime**, not as a single-variable model. This is what lets the model separate "does Animation earn more because it's Animation" from "does Animation earn more because Animation movies tend to have bigger budgets," a distinction a simple genre-only comparison can't make.
- **Runtime turns out not to be a significant predictor** (p = 0.82 in the fitted model) once budget, critic score, and genre are accounted for. That's reported plainly rather than dropped from the writeup, since a non-significant result is still a real finding, not a failed one.

## Running it locally

```bash
pip install pandas numpy scipy statsmodels

python3 data_generation.py              # generates data/movies.csv
python3 scripts/inferential_stats.py    # confidence intervals, ANOVA, t-test, regression
```

## Stack

Python, pandas, NumPy, SciPy (hypothesis testing), statsmodels (ANOVA via formula API, OLS multilinear regression).
