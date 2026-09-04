import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from db.models import Interaction, Course

class CollaborativeFilteringRecommender:
    def __init__(self, n_factors: int = 10, min_interactions: int = 3, model_path: Optional[str] = None):
        self.n_factors = n_factors
        self.min_interactions = min_interactions
        self.model_path = model_path
        self.svd = None
        self.learner_ids = []
        self.course_ids = []
        self.learner_id_to_idx = {}
        self.course_id_to_idx = {}
        self.predicted_ratings_matrix = None
        self.global_mean = 3.0
        self.is_fitted = False

    def fit(self, interactions: List[Interaction], all_courses: List[Course]):
        """
        Builds user-item interaction matrix from ratings and completion signals,
        then factorizes using SVD.
        """
        if not interactions:
            return self

        records = []
        for it in interactions:
            # Composite implicit-explicit signal: rating (1-5) or quiz_score/20, completion bonus
            val = it.rating if it.rating > 0 else (it.quiz_score / 20.0 if it.quiz_score > 0 else 3.0)
            if it.completed:
                val = min(5.0, val + 0.5)
            records.append({
                "learner_id": it.learner_id,
                "course_id": it.course_id,
                "rating": val
            })

        df = pd.DataFrame(records)
        self.global_mean = float(df["rating"].mean()) if not df.empty else 3.0

        # Create pivot table
        pivot_df = df.pivot_table(index="learner_id", columns="course_id", values="rating")
        
        # Ensure all courses are columns
        for c in all_courses:
            if c.course_id not in pivot_df.columns:
                pivot_df[c.course_id] = np.nan

        self.learner_ids = list(pivot_df.index)
        self.course_ids = list(pivot_df.columns)
        self.learner_id_to_idx = {lid: idx for idx, lid in enumerate(self.learner_ids)}
        self.course_id_to_idx = {cid: idx for idx, cid in enumerate(self.course_ids)}

        # Fill missing values with user mean or global mean
        user_means = pivot_df.mean(axis=1).fillna(self.global_mean)
        imputed_matrix = pivot_df.apply(lambda row: row.fillna(user_means[row.name]), axis=1).values

        n_components = min(self.n_factors, min(imputed_matrix.shape) - 1)
        if n_components >= 1:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            u_sigma = self.svd.fit_transform(imputed_matrix)
            vt = self.svd.components_
            self.predicted_ratings_matrix = np.dot(u_sigma, vt)
        else:
            self.predicted_ratings_matrix = imputed_matrix

        self.is_fitted = True

        if self.model_path:
            p = Path(self.model_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump((
                    self.svd,
                    self.learner_ids,
                    self.course_ids,
                    self.learner_id_to_idx,
                    self.course_id_to_idx,
                    self.predicted_ratings_matrix,
                    self.global_mean
                ), f)

        return self

    def load(self, model_path: str):
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                (
                    self.svd,
                    self.learner_ids,
                    self.course_ids,
                    self.learner_id_to_idx,
                    self.course_id_to_idx,
                    self.predicted_ratings_matrix,
                    self.global_mean
                ) = pickle.load(f)
                self.is_fitted = True
        return self

    def is_cold_start(self, learner_id: str, interactions: List[Interaction]) -> bool:
        learner_ints = [it for it in interactions if it.learner_id == learner_id]
        return len(learner_ints) < self.min_interactions

    def score(self, learner_id: str, courses: List[Course], interactions: List[Interaction]) -> Dict[str, float]:
        """
        Predicts collaborative filtering score for each course in [0.0, 1.0].
        If learner is cold-start, returns neutral baseline (0.5).
        """
        if not self.is_fitted:
            self.fit(interactions, courses)

        if self.is_cold_start(learner_id, interactions) or learner_id not in self.learner_id_to_idx:
            # Cold start fallback
            return {c.course_id: 0.5 for c in courses}

        u_idx = self.learner_id_to_idx[learner_id]
        user_preds = self.predicted_ratings_matrix[u_idx]

        scores = {}
        for c in courses:
            if c.course_id in self.course_id_to_idx:
                c_idx = self.course_id_to_idx[c.course_id]
                pred_rating = float(user_preds[c_idx])
                # Normalize 1.0 - 5.0 scale to 0.0 - 1.0
                norm_score = max(0.0, min(1.0, (pred_rating - 1.0) / 4.0))
                scores[c.course_id] = round(norm_score, 4)
            else:
                scores[c.course_id] = 0.5

        return scores
