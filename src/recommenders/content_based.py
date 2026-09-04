import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from db.models import Learner, Course, Interaction
from src.catalog.catalog_service import CatalogService

class ContentBasedRecommender:
    def __init__(self, vectorizer_path: Optional[str] = None):
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        self.course_vectors = None
        self.course_id_to_idx = {}
        self.idx_to_course_id = {}
        self.is_fitted = False

    def fit(self, courses: List[Course]):
        """
        Fits TF-IDF vectorizer on course representations (name, description, tags, skills).
        """
        if not courses:
            return self

        corpus = [CatalogService.get_course_representation_text(c) for c in courses]
        self.course_vectors = self.vectorizer.fit_transform(corpus).toarray()
        
        self.course_id_to_idx = {c.course_id: idx for idx, c in enumerate(courses)}
        self.idx_to_course_id = {idx: c.course_id for idx, c in enumerate(courses)}
        self.is_fitted = True

        if self.vectorizer_path:
            p = Path(self.vectorizer_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump((self.vectorizer, self.course_vectors, self.course_id_to_idx, self.idx_to_course_id), f)

        return self

    def load(self, vectorizer_path: str):
        if Path(vectorizer_path).exists():
            with open(vectorizer_path, "rb") as f:
                self.vectorizer, self.course_vectors, self.course_id_to_idx, self.idx_to_course_id = pickle.load(f)
                self.is_fitted = True
        return self

    def build_learner_vector(self, learner: Learner, interactions: List[Interaction]) -> np.ndarray:
        """
        Creates learner profile representation combining stated interests/skills
        and weighted average of completed/rated courses.
        """
        # Stated profile text
        interests = " ".join(learner.get_interests())
        skills = " ".join(learner.get_skills())
        pref_topics = " ".join(learner.get_preferred_topics())
        profile_text = f"{interests} {skills} {pref_topics}"

        profile_vector = self.vectorizer.transform([profile_text]).toarray()[0]

        # Interacted courses vectors weighted by rating / completion
        interaction_vectors = []
        weights = []
        for it in interactions:
            if it.course_id in self.course_id_to_idx:
                c_idx = self.course_id_to_idx[it.course_id]
                c_vec = self.course_vectors[c_idx]
                
                # Weight: higher for completed and high ratings
                w = 1.0
                if it.completed:
                    w += 1.0
                if it.rating > 0:
                    w += (it.rating / 5.0)
                
                interaction_vectors.append(c_vec)
                weights.append(w)

        if interaction_vectors:
            history_vector = np.average(interaction_vectors, axis=0, weights=weights)
            # Combine stated profile (40%) and behavioral history (60%)
            combined = 0.4 * profile_vector + 0.6 * history_vector
        else:
            combined = profile_vector

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    def score(self, learner: Learner, courses: List[Course], interactions: List[Interaction]) -> Dict[str, float]:
        """
        Computes cosine similarity between learner vector and each course vector.
        Returns a dict of {course_id: score} bounded in [0.0, 1.0].
        """
        if not self.is_fitted:
            self.fit(courses)

        learner_vec = self.build_learner_vector(learner, interactions).reshape(1, -1)
        sims = cosine_similarity(learner_vec, self.course_vectors)[0]

        scores = {}
        for c in courses:
            if c.course_id in self.course_id_to_idx:
                idx = self.course_id_to_idx[c.course_id]
                sim = float(sims[idx])
                # Normalize from [-1, 1] to [0, 1]
                norm_sim = max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))
                scores[c.course_id] = round(norm_sim, 4)
            else:
                scores[c.course_id] = 0.5

        return scores
