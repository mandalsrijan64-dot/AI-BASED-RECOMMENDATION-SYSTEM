"""
app.py  –  REST API for the Recommendation System
==================================================
Endpoints:
  GET  /users                        – list all users
  GET  /movies                       – list all movies
  GET  /ratings/<user_id>            – ratings for a user
  GET  /recommend/<user_id>          – hybrid recommendations (default)
  GET  /recommend/<user_id>?method=cb|cf|hybrid&top_n=5
  GET  /similar-users/<user_id>      – most similar users
"""

from flask import Flask, jsonify, request
from recommender import (
    USERS, MOVIES, RATINGS,
    content_based_recommend,
    collaborative_recommend,
    hybrid_recommend,
    similar_users,
)

app = Flask(__name__)


@app.route("/users")
def get_users():
    return jsonify({"users": USERS})


@app.route("/movies")
def get_movies():
    return jsonify({
        "movies": [
            {"id": mid, **meta}
            for mid, meta in MOVIES.items()
        ]
    })


@app.route("/ratings/<user_id>")
def get_ratings(user_id: str):
    if user_id not in RATINGS:
        return jsonify({"error": "User not found"}), 404
    ratings = [
        {
            "movie_id": mid,
            "title": MOVIES[mid]["title"],
            "rating": r,
        }
        for mid, r in RATINGS[user_id].items()
    ]
    return jsonify({"user": user_id, "ratings": ratings})


@app.route("/recommend/<user_id>")
def recommend(user_id: str):
    if user_id not in RATINGS:
        return jsonify({"error": "User not found"}), 404

    method = request.args.get("method", "hybrid").lower()
    top_n  = int(request.args.get("top_n", 5))

    if method == "cb":
        recs = content_based_recommend(user_id, top_n=top_n)
    elif method == "cf":
        recs = collaborative_recommend(user_id, top_n=top_n)
    else:
        recs = hybrid_recommend(user_id, top_n=top_n)

    return jsonify({
        "user": user_id,
        "method": method,
        "recommendations": recs,
    })


@app.route("/similar-users/<user_id>")
def get_similar_users(user_id: str):
    if user_id not in RATINGS:
        return jsonify({"error": "User not found"}), 404
    top_n = int(request.args.get("top_n", 3))
    return jsonify({
        "user": user_id,
        "similar_users": similar_users(user_id, top_n=top_n),
    })


if __name__ == "__main__":
    print("Starting Recommendation API on http://localhost:5000")
    app.run(debug=True, port=5000)
