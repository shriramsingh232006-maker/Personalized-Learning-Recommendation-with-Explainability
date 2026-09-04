import os
try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from db.models import Learner, Course, Interaction, SkillProfile, RecommendationLog
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative import CollaborativeFilteringRecommender
from src.recommenders.performance_score import PerformanceScorer
from src.explainability.rule_based_explainer import RuleBasedExplainer
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LimeExplainer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

class HybridEngine:
    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        top_k: int = 10,
        min_interactions_for_cf: int = 3
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k
        self.min_interactions_for_cf = min_interactions_for_cf

        # Load from config if available
        self._load_config()

        self.cb_recommender = ContentBasedRecommender(
            vectorizer_path=str(BASE_DIR / "models" / "content_vectorizer.pkl")
        )
        self.cf_recommender = CollaborativeFilteringRecommender(
            n_factors=10,
            min_interactions=self.min_interactions_for_cf,
            model_path=str(BASE_DIR / "models" / "cf_model.pkl")
        )
        self.perf_scorer = PerformanceScorer()
        self.shap_explainer = ShapExplainer()
        self.lime_explainer = LimeExplainer()

    def _load_config(self):
        if yaml and CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg and "recommender" in cfg:
                        weights = cfg["recommender"].get("weights", {})
                        self.alpha = float(weights.get("alpha", self.alpha))
                        self.beta = float(weights.get("beta", self.beta))
                        self.gamma = float(weights.get("gamma", self.gamma))
                        self.top_k = int(cfg["recommender"].get("top_k", self.top_k))
                        self.min_interactions_for_cf = int(cfg["recommender"].get("min_interactions_for_cf", self.min_interactions_for_cf))
            except Exception as e:
                print(f"Warning: Could not parse config.yaml: {e}")

    def train_models(self, db: Session):
        """
        Trains or fits both Content-Based and Collaborative Filtering models on current DB data.
        """
        courses = db.query(Course).all()
        interactions = db.query(Interaction).all()
        self.cb_recommender.fit(courses)
        self.cf_recommender.fit(interactions, courses)

    def recommend(
        self,
        db: Session,
        learner_id: str,
        k: Optional[int] = None,
        filter_completed: bool = True,
        log_recommendations: bool = False
    ) -> List[Dict[str, Any]]:
        k = k or self.top_k
        learner = db.query(Learner).filter(Learner.learner_id == learner_id).first()
        if not learner:
            raise ValueError(f"Learner '{learner_id}' not found.")

        all_courses = db.query(Course).all()
        if not all_courses:
            return []

        all_interactions = db.query(Interaction).all()
        learner_interactions = [it for it in all_interactions if it.learner_id == learner_id]
        skill_profiles = db.query(SkillProfile).filter(SkillProfile.learner_id == learner_id).all()

        # Check cold-start condition
        is_cold = len(learner_interactions) < self.min_interactions_for_cf
        if is_cold:
            w_alpha = 0.7
            w_beta = 0.0
            w_gamma = 0.3
        else:
            w_alpha = self.alpha
            w_beta = self.beta
            w_gamma = self.gamma

        # Normalize weights to sum to 1.0
        total_w = w_alpha + w_beta + w_gamma
        w_alpha /= total_w
        w_beta /= total_w
        w_gamma /= total_w

        # Compute individual module scores
        cb_scores = self.cb_recommender.score(learner, all_courses, learner_interactions)
        cf_scores = self.cf_recommender.score(learner_id, all_courses, all_interactions)
        perf_scores = self.perf_scorer.score(learner, all_courses, learner_interactions, skill_profiles)

        completed_ids = {it.course_id for it in learner_interactions if it.completed}

        results = []
        for course in all_courses:
            cid = course.course_id
            if filter_completed and cid in completed_ids:
                continue

            s_cb = cb_scores.get(cid, 0.5)
            s_cf = cf_scores.get(cid, 0.5)
            s_perf = perf_scores.get(cid, 0.5)

            hybrid_score = (w_alpha * s_cb) + (w_beta * s_cf) + (w_gamma * s_perf)
            hybrid_score = round(float(hybrid_score), 4)

            score_breakdown = {
                "content_score": s_cb,
                "collaborative_score": s_cf,
                "performance_score": s_perf,
                "weight_content": w_alpha,
                "weight_collaborative": w_beta,
                "weight_performance": w_gamma,
                "hybrid_score": hybrid_score
            }

            # Generate Rule-based explanation
            rule_xai = RuleBasedExplainer.explain(
                learner=learner,
                course=course,
                score_breakdown=score_breakdown,
                interactions=learner_interactions,
                skill_profiles=skill_profiles
            )

            # Generate SHAP attribution
            shap_info = self.shap_explainer.explain_instance(s_cb, s_cf, s_perf)

            results.append({
                "course_id": course.course_id,
                "name": course.name,
                "topic": course.topic,
                "difficulty": course.difficulty,
                "duration": course.duration,
                "hybrid_score": hybrid_score,
                "content_score": s_cb,
                "collaborative_score": s_cf,
                "performance_score": s_perf,
                "is_cold_start": is_cold,
                "explanation_text": rule_xai["explanation_text"],
                "primary_factor": rule_xai["primary_factor"],
                "factor_contributions": rule_xai["factor_contributions"],
                "shap_values": shap_info["shap_values"],
                "tags": course.get_tags(),
                "skills_covered": course.get_skills_covered()
            })

        # Rank by hybrid score descending
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_recommendations = results[:k]

        # Optional logging
        if log_recommendations:
            for rec in top_recommendations:
                log_entry = RecommendationLog(
                    learner_id=learner_id,
                    course_id=rec["course_id"],
                    hybrid_score=rec["hybrid_score"],
                    explanation_text=rec["explanation_text"]
                )
                db.add(log_entry)
            db.commit()

        return top_recommendations
