"""
Recommendation System
=====================
Implements two filtering strategies:
  1. Content-Based Filtering  – recommends items similar to what a user liked
  2. Collaborative Filtering  – recommends items liked by similar users (user-user)
  3. Hybrid                   – weighted blend of both scores
"""

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Sample dataset: users, movies, ratings (1-5), movie metadata
# ---------------------------------------------------------------------------

MOVIES: dict[int, dict] = {
    1:  {"title": "Inception",          "genres": ["Sci-Fi", "Thriller"],        "year": 2010, "director": "Christopher Nolan"},
    2:  {"title": "The Dark Knight",    "genres": ["Action", "Thriller"],        "year": 2008, "director": "Christopher Nolan"},
    3:  {"title": "Interstellar",       "genres": ["Sci-Fi", "Drama"],           "year": 2014, "director": "Christopher Nolan"},
    4:  {"title": "The Matrix",         "genres": ["Sci-Fi", "Action"],          "year": 1999, "director": "Wachowski Sisters"},
    5:  {"title": "Parasite",           "genres": ["Thriller", "Drama"],         "year": 2019, "director": "Bong Joon-ho"},
    6:  {"title": "Avengers: Endgame",  "genres": ["Action", "Sci-Fi"],          "year": 2019, "director": "Russo Brothers"},
    7:  {"title": "The Godfather",      "genres": ["Crime", "Drama"],            "year": 1972, "director": "Francis Ford Coppola"},
    8:  {"title": "Pulp Fiction",       "genres": ["Crime", "Thriller"],         "year": 1994, "director": "Quentin Tarantino"},
    9:  {"title": "Spirited Away",      "genres": ["Animation", "Fantasy"],      "year": 2001, "director": "Hayao Miyazaki"},
    10: {"title": "The Prestige",       "genres": ["Thriller", "Drama"],         "year": 2006, "director": "Christopher Nolan"},
    11: {"title": "Blade Runner 2049",  "genres": ["Sci-Fi", "Drama"],           "year": 2017, "director": "Denis Villeneuve"},
    12: {"title": "Arrival",            "genres": ["Sci-Fi", "Drama"],           "year": 2016, "director": "Denis Villeneuve"},
    13: {"title": "Fight Club",         "genres": ["Drama", "Thriller"],         "year": 1999, "director": "David Fincher"},
    14: {"title": "Princess Mononoke",  "genres": ["Animation", "Fantasy"],      "year": 1997, "director": "Hayao Miyazaki"},
    15: {"title": "Goodfellas",         "genres": ["Crime", "Drama"],            "year": 1990, "director": "Martin Scorsese"},
}

# user_id -> {movie_id -> rating}
RATINGS: dict[str, dict[int, float]] = {
    "alice":   {1: 5, 2: 4, 3: 5, 4: 4, 10: 4, 11: 3},
    "bob":     {4: 5, 6: 4, 2: 3, 11: 5, 12: 4, 3: 3},
    "carol":   {7: 5, 8: 5, 15: 4, 5: 4, 13: 3},
    "dave":    {1: 4, 3: 5, 11: 5, 12: 5, 4: 3},
    "eve":     {9: 5, 14: 5, 5: 4, 7: 3},
    "frank":   {2: 5, 6: 5, 4: 4, 1: 3, 8: 4},
    "grace":   {5: 5, 7: 4, 13: 5, 8: 3, 15: 4},
    "henry":   {9: 4, 14: 5, 3: 3, 12: 4},
    "iris":    {1: 5, 10: 5, 2: 4, 13: 3, 8: 4},
    "jack":    {6: 5, 4: 5, 2: 4, 11: 4},
}

USERS = list(RATINGS.keys())


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pearson_similarity(vec_a: dict, vec_b: dict) -> float:
    """Pearson correlation between two sparse rating vectors."""
    common = set(vec_a) & set(vec_b)
    n = len(common)
    if n < 2:
        return 0.0
    mean_a = sum(vec_a[k] for k in common) / n
    mean_b = sum(vec_b[k] for k in common) / n
    num = sum((vec_a[k] - mean_a) * (vec_b[k] - mean_b) for k in common)
    den_a = math.sqrt(sum((vec_a[k] - mean_a) ** 2 for k in common))
    den_b = math.sqrt(sum((vec_b[k] - mean_b) ** 2 for k in common))
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def _movie_feature_vector(movie_id: int) -> dict:
    """Binary feature vector: genre flags + director flag."""
    movie = MOVIES[movie_id]
    vec = {}
    for genre in movie["genres"]:
        vec[f"genre_{genre}"] = 1.0
    vec[f"director_{movie['director']}"] = 1.0
    return vec


# ---------------------------------------------------------------------------
# 1. Content-Based Filtering
# ---------------------------------------------------------------------------

def content_based_recommend(
    user_id: str,
    top_n: int = 5,
) -> list[dict]:
    """
    Build a profile from the user's rated movies, then score all unseen
    movies by cosine similarity to that profile.
    """
    rated = RATINGS.get(user_id, {})
    if not rated:
        return []

    # Build weighted user-profile vector
    profile: dict[str, float] = {}
    for mid, rating in rated.items():
        weight = rating / 5.0          # normalise to [0, 1]
        for feat, val in _movie_feature_vector(mid).items():
            profile[feat] = profile.get(feat, 0.0) + weight * val

    # Score unseen movies
    candidates = [mid for mid in MOVIES if mid not in rated]
    scores = []
    for mid in candidates:
        fvec = _movie_feature_vector(mid)
        sim = _cosine_similarity(profile, fvec)
        scores.append((mid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, score in scores[:top_n]:
        results.append({
            "movie_id": mid,
            "title": MOVIES[mid]["title"],
            "genres": MOVIES[mid]["genres"],
            "year": MOVIES[mid]["year"],
            "score": round(score, 4),
            "method": "content-based",
        })
    return results


# ---------------------------------------------------------------------------
# 2. Collaborative Filtering (User-User)
# ---------------------------------------------------------------------------

def collaborative_recommend(
    user_id: str,
    top_n: int = 5,
    k_neighbors: int = 4,
) -> list[dict]:
    """
    Find the k most similar users (Pearson), then predict ratings for
    unseen movies using weighted average of neighbor ratings.
    """
    user_ratings = RATINGS.get(user_id, {})

    # Compute similarity with every other user
    similarities = []
    for other_id, other_ratings in RATINGS.items():
        if other_id == user_id:
            continue
        sim = _pearson_similarity(user_ratings, other_ratings)
        if sim > 0:
            similarities.append((other_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    neighbors = similarities[:k_neighbors]

    if not neighbors:
        return []

    # Predict score for each unseen movie
    candidates = [mid for mid in MOVIES if mid not in user_ratings]
    predictions = []
    for mid in candidates:
        num, den = 0.0, 0.0
        for neighbor_id, sim in neighbors:
            if mid in RATINGS[neighbor_id]:
                num += sim * RATINGS[neighbor_id][mid]
                den += abs(sim)
        if den > 0:
            predictions.append((mid, num / den))

    predictions.sort(key=lambda x: x[1], reverse=True)

    results = []
    for mid, pred_rating in predictions[:top_n]:
        results.append({
            "movie_id": mid,
            "title": MOVIES[mid]["title"],
            "genres": MOVIES[mid]["genres"],
            "year": MOVIES[mid]["year"],
            "score": round(pred_rating, 4),
            "method": "collaborative",
        })
    return results


# ---------------------------------------------------------------------------
# 3. Hybrid Recommender
# ---------------------------------------------------------------------------

def hybrid_recommend(
    user_id: str,
    top_n: int = 5,
    cb_weight: float = 0.4,
    cf_weight: float = 0.6,
) -> list[dict]:
    """
    Normalise scores from both methods to [0,1] then blend.
    """
    cb = content_based_recommend(user_id, top_n=len(MOVIES))
    cf = collaborative_recommend(user_id, top_n=len(MOVIES))

    def _normalise(recs: list[dict], key: str = "score") -> dict[int, float]:
        if not recs:
            return {}
        vals = [r[key] for r in recs]
        mn, mx = min(vals), max(vals)
        rng = mx - mn or 1e-9
        return {r["movie_id"]: (r[key] - mn) / rng for r in recs}

    cb_norm = _normalise(cb)
    cf_norm = _normalise(cf)

    all_ids = set(cb_norm) | set(cf_norm)
    blended = []
    for mid in all_ids:
        score = cb_weight * cb_norm.get(mid, 0) + cf_weight * cf_norm.get(mid, 0)
        blended.append({
            "movie_id": mid,
            "title": MOVIES[mid]["title"],
            "genres": MOVIES[mid]["genres"],
            "year": MOVIES[mid]["year"],
            "score": round(score, 4),
            "method": "hybrid",
        })

    blended.sort(key=lambda x: x["score"], reverse=True)
    return blended[:top_n]


# ---------------------------------------------------------------------------
# 4. User similarity report
# ---------------------------------------------------------------------------

def similar_users(user_id: str, top_n: int = 3) -> list[dict]:
    user_ratings = RATINGS.get(user_id, {})
    sims = []
    for other_id, other_ratings in RATINGS.items():
        if other_id == user_id:
            continue
        sim = _pearson_similarity(user_ratings, other_ratings)
        sims.append({"user": other_id, "similarity": round(sim, 4)})
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return sims[:top_n]


# ---------------------------------------------------------------------------
# CLI Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TARGET = "alice"
    print(f"\n{'='*60}")
    print(f"  Recommendation Demo  |  User: {TARGET.upper()}")
    print(f"{'='*60}")

    print(f"\nRatings by {TARGET}:")
    for mid, r in RATINGS[TARGET].items():
        print(f"  {'★'*r}{'☆'*(5-r)}  {MOVIES[mid]['title']}")

    print(f"\n── Similar Users ──")
    for u in similar_users(TARGET):
        print(f"  {u['user']:10s}  sim={u['similarity']:.3f}")

    for method, fn in [
        ("Content-Based", content_based_recommend),
        ("Collaborative", collaborative_recommend),
        ("Hybrid",        hybrid_recommend),
    ]:
        print(f"\n── {method} Recommendations ──")
        for i, rec in enumerate(fn(TARGET, top_n=5), 1):
            genres = ", ".join(rec["genres"])
            print(f"  {i}. {rec['title']} ({rec['year']})  [{genres}]  score={rec['score']}")
