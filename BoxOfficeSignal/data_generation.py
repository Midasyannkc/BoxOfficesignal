"""
Generates a synthetic movie dataset for inferential statistics: one row per
film, with genre, budget, box office revenue, critic score, runtime, and
release season. Revenue is built from a genre-specific multiplier and a
budget relationship with real noise, so the confidence intervals and
hypothesis tests run downstream have genuine signal to detect, not pure
randomness.
"""

import csv
import random
from pathlib import Path

random.seed(99)

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENRES = ["Action", "Comedy", "Drama", "Horror", "Animation", "Sci-Fi", "Romance", "Thriller"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

GENRE_ROI_MULTIPLIER = {
    "Action": 2.6, "Comedy": 2.2, "Drama": 1.5, "Horror": 4.8,
    "Animation": 3.1, "Sci-Fi": 2.4, "Romance": 1.8, "Thriller": 2.0,
}

GENRE_BUDGET_RANGE = {
    "Action": (60, 220), "Comedy": (20, 90), "Drama": (10, 60), "Horror": (5, 40),
    "Animation": (70, 200), "Sci-Fi": (50, 200), "Romance": (10, 50), "Thriller": (20, 80),
}


def sample_critic_score(genre):
    base = 58
    if genre in ("Drama", "Animation"):
        base += 8
    if genre == "Horror":
        base -= 5
    return round(min(99, max(10, random.gauss(base, 14))), 1)


def sample_runtime(genre):
    if genre == "Animation":
        return round(random.gauss(98, 8))
    if genre == "Drama":
        return round(random.gauss(122, 15))
    return round(random.gauss(112, 13))


def generate_movies(n=600):
    rows = []
    for i in range(1, n + 1):
        genre = random.choice(GENRES)
        budget_low, budget_high = GENRE_BUDGET_RANGE[genre]
        budget_millions = round(random.uniform(budget_low, budget_high), 1)

        critic_score = sample_critic_score(genre)
        runtime = sample_runtime(genre)
        season = random.choices(SEASONS, weights=[0.20, 0.20, 0.35, 0.25])[0]

        base_multiplier = GENRE_ROI_MULTIPLIER[genre]
        quality_adjustment = 1.0 + (critic_score - 58) / 200
        season_adjustment = 1.15 if season == "Summer" else 1.0

        noise = random.lognormvariate(0, 0.45)
        revenue_millions = round(
            budget_millions * base_multiplier * quality_adjustment * season_adjustment * noise, 1
        )

        rows.append({
            "movie_id": f"MV{i:04d}",
            "genre": genre,
            "budget_millions": budget_millions,
            "revenue_millions": revenue_millions,
            "critic_score": critic_score,
            "runtime_minutes": runtime,
            "release_season": season,
        })
    return rows


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    movies = generate_movies()
    write_csv(
        movies,
        OUTPUT_DIR / "movies.csv",
        ["movie_id", "genre", "budget_millions", "revenue_millions",
         "critic_score", "runtime_minutes", "release_season"],
    )
