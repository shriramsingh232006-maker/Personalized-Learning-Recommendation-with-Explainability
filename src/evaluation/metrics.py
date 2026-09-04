import math
from typing import List, Dict, Set, Any
from sqlalchemy.orm import Session
import numpy as np

from db.models import Interaction, Course

class RecommendationEvaluator:
    @staticmethod
    def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if k == 0 or not recommended:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return hits / float(k)

    @staticmethod
    def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant:
            return 0.0
        top_k = recommended[:k]
        hits = sum(1 for item in top_k if item in relevant)
        return hits / float(len(relevant))

    @staticmethod
    def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant or not recommended:
            return 0.0
        top_k = recommended[:k]
        dcg = 0.0
        for i, item in enumerate(top_k):
            if item in relevant:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because 1-based index (log2(1+1) = 1)

        # Ideal DCG
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def average_precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
        if not relevant or not recommended:
            return 0.0
        top_k = recommended[:k]
        hits = 0
        sum_precisions = 0.0
        for i, item in enumerate(top_k):
            if item in relevant:
                hits += 1
                sum_precisions += hits / float(i + 1)
        if hits == 0:
            return 0.0
        return sum_precisions / float(min(len(relevant), k))

    @staticmethod
    def evaluate_offline(
        db: Session,
        hybrid_engine: Any,
        k: int = 5,
        rating_threshold: float = 3.5
    ) -> Dict[str, Any]:
        """
        Evaluates the recommendation engine on held-out positive interactions
        (ratings >= 3.5 or completed == True).
        """
        interactions = db.query(Interaction).all()
        learners_map: Dict[str, Set[str]] = {}

        for it in interactions:
            if it.completed or it.rating >= rating_threshold or it.quiz_score >= 70.0:
                if it.learner_id not in learners_map:
                    learners_map[it.learner_id] = set()
                learners_map[it.learner_id].add(it.course_id)

        precisions = []
        recalls = []
        ndcgs = []
        maps = []

        eval_learners = [lid for lid, rel in learners_map.items() if len(rel) >= 2]
        
        for lid in eval_learners:
            relevant = learners_map[lid]
            try:
                recs = hybrid_engine.recommend(db=db, learner_id=lid, k=k, filter_completed=False)
                rec_ids = [r["course_id"] for r in recs]

                p = RecommendationEvaluator.precision_at_k(rec_ids, relevant, k)
                r = RecommendationEvaluator.recall_at_k(rec_ids, relevant, k)
                ndcg = RecommendationEvaluator.ndcg_at_k(rec_ids, relevant, k)
                ap = RecommendationEvaluator.average_precision_at_k(rec_ids, relevant, k)

                precisions.append(p)
                recalls.append(r)
                ndcgs.append(ndcg)
                maps.append(ap)
            except Exception:
                continue

        return {
            "k": k,
            "total_evaluated_learners": len(precisions),
            "precision_at_k": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "recall_at_k": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
            "map_at_k": round(float(np.mean(maps)), 4) if maps else 0.0
        }
