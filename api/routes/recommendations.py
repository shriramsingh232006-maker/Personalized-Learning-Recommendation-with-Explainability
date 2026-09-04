import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.database import get_db
from src.recommenders.hybrid_engine import HybridEngine
from api.schemas.recommendation import RecommendationResponse, RecommendationItem

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

# Singleton engine instance
engine = HybridEngine()

@router.get("/{learner_id}", response_model=RecommendationResponse)
def get_recommendations(
    learner_id: str,
    k: int = Query(10, ge=1, le=50, description="Number of recommendations to return"),
    filter_completed: bool = Query(True, description="Filter out courses already completed by learner"),
    db: Session = Depends(get_db)
):
    """
    Generate personalized hybrid recommendations with explainability for a learner.
    Combines Content-Based (CB), Collaborative Filtering (CF), and Performance/Difficulty alignment (PS).
    """
    try:
        recommendations = engine.recommend(
            db=db,
            learner_id=learner_id,
            k=k,
            filter_completed=filter_completed,
            log_recommendations=True
        )
        return RecommendationResponse(
            learner_id=learner_id,
            k=k,
            recommendations=[RecommendationItem(**r) for r in recommendations]
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {str(e)}")
