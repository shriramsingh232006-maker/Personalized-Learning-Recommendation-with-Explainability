import sys
from pathlib import Path
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.database import get_db
from src.evaluation.metrics import RecommendationEvaluator
from src.recommenders.hybrid_engine import HybridEngine
from api.schemas.recommendation import EvaluationMetricsResponse

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
engine = HybridEngine()

@router.get("/metrics", response_model=EvaluationMetricsResponse)
def get_evaluation_metrics(
    k: int = Query(5, ge=1, le=20, description="Top-K cutoff for evaluation metrics"),
    db: Session = Depends(get_db)
):
    """
    Compute Precision@K, Recall@K, NDCG@K, and MAP@K on interaction data.
    """
    metrics = RecommendationEvaluator.evaluate_offline(db=db, hybrid_engine=engine, k=k)
    return EvaluationMetricsResponse(**metrics)
