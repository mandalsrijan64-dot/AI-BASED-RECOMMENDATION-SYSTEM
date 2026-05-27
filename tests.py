"""
tests.py  –  Unit tests for the Recommendation System
"""

import math
import unittest
from recommender import (
    MOVIES, RATINGS, USERS,
    _cosine_similarity,
    _pearson_similarity,
    _movie_feature_vector,
    content_based_recommend,
    collaborative_recommend,
    hybrid_recommend,
    similar_users,
)


class TestSimilarityFunctions(unittest.TestCase):

    def test_cosine_identical_vectors(self):
        vec = {"a": 1, "b": 2, "c": 3}
        self.assertAlmostEqual(_cosine_similarity(vec, vec), 1.0)

    def test_cosine_orthogonal_vectors(self):
        self.assertAlmostEqual(_cosine_similarity({"a": 1}, {"b": 1}), 0.0)

    def test_cosine_empty(self):
        self.assertEqual(_cosine_similarity({}, {"a": 1}), 0.0)

    def test_pearson_identical(self):
        vec = {1: 4, 2: 2, 3: 5}
        self.assertAlmostEqual(_pearson_similarity(vec, vec), 1.0)

    def test_pearson_no_common(self):
        self.assertEqual(_pearson_similarity({1: 5}, {2: 5}), 0.0)

    def test_pearson_single_common(self):
        # Only 1 common item – return 0 (undefined)
        self.assertEqual(_pearson_similarity({1: 5}, {1: 5}), 0.0)


class TestFeatureVector(unittest.TestCase):

    def test_inception_features(self):
        vec = _movie_feature_vector(1)   # Inception: Sci-Fi, Thriller, Nolan
        self.assertIn("genre_Sci-Fi", vec)
        self.assertIn("genre_Thriller", vec)
        self.assertIn("director_Christopher Nolan", vec)

    def test_feature_values_are_positive(self):
        for mid in MOVIES:
            vec = _movie_feature_vector(mid)
            for v in vec.values():
                self.assertGreater(v, 0)


class TestContentBased(unittest.TestCase):

    def test_returns_list(self):
        recs = content_based_recommend("alice")
        self.assertIsInstance(recs, list)

    def test_top_n_respected(self):
        recs = content_based_recommend("alice", top_n=3)
        self.assertLessEqual(len(recs), 3)

    def test_no_already_rated(self):
        rated = set(RATINGS["alice"].keys())
        recs = content_based_recommend("alice", top_n=10)
        rec_ids = {r["movie_id"] for r in recs}
        self.assertTrue(rec_ids.isdisjoint(rated))

    def test_unknown_user_returns_empty(self):
        recs = content_based_recommend("nonexistent_user")
        self.assertEqual(recs, [])

    def test_scores_in_range(self):
        recs = content_based_recommend("bob", top_n=10)
        for r in recs:
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.01)    # slight float tolerance


class TestCollaborative(unittest.TestCase):

    def test_returns_list(self):
        recs = collaborative_recommend("alice")
        self.assertIsInstance(recs, list)

    def test_top_n_respected(self):
        recs = collaborative_recommend("carol", top_n=4)
        self.assertLessEqual(len(recs), 4)

    def test_no_already_rated(self):
        rated = set(RATINGS["carol"].keys())
        recs = collaborative_recommend("carol", top_n=10)
        rec_ids = {r["movie_id"] for r in recs}
        self.assertTrue(rec_ids.isdisjoint(rated))

    def test_method_label(self):
        recs = collaborative_recommend("dave", top_n=3)
        for r in recs:
            self.assertEqual(r["method"], "collaborative")


class TestHybrid(unittest.TestCase):

    def test_returns_list(self):
        recs = hybrid_recommend("iris")
        self.assertIsInstance(recs, list)

    def test_scores_between_0_and_1(self):
        recs = hybrid_recommend("frank", top_n=10)
        for r in recs:
            self.assertGreaterEqual(r["score"], -0.01)
            self.assertLessEqual(r["score"], 1.01)

    def test_no_already_rated(self):
        rated = set(RATINGS["grace"].keys())
        recs = hybrid_recommend("grace", top_n=10)
        rec_ids = {r["movie_id"] for r in recs}
        self.assertTrue(rec_ids.isdisjoint(rated))


class TestSimilarUsers(unittest.TestCase):

    def test_returns_list(self):
        result = similar_users("alice")
        self.assertIsInstance(result, list)

    def test_excludes_self(self):
        result = similar_users("alice")
        user_ids = [r["user"] for r in result]
        self.assertNotIn("alice", user_ids)

    def test_top_n_respected(self):
        result = similar_users("bob", top_n=2)
        self.assertLessEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
