import sys
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.database import get_db
from src.feedback.feedback_service import FeedbackService
from api.schemas.recommendation import FeedbackCreate, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    """
    Submit user feedback (course rating, quiz score, or completion status).
    This updates interactions and dynamically recalculates the learner's topic skill profile.
    """
    try:
        result = FeedbackService.submit_feedback(
            db=db,
            learner_id=payload.learner_id,
            course_id=payload.course_id,
            rating=payload.rating,
            quiz_score=payload.quiz_score,
            completed=payload.completed
        )
        return FeedbackResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission error: {str(e)}")
