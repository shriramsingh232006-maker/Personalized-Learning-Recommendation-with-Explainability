import sys
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.database import get_db
from src.profile.profile_service import ProfileService
from api.schemas.learner import LearnerResponse, LearnerProfileResponse, LearnerCreate

router = APIRouter(prefix="/learners", tags=["Learners"])

@router.get("/", response_model=List[LearnerResponse])
def list_learners(db: Session = Depends(get_db)):
    """Fetch all learners."""
    learners = ProfileService.get_all_learners(db)
    return [
        LearnerResponse(
            learner_id=l.learner_id,
            name=l.name,
            interests=l.get_interests(),
            education_background=l.education_background,
            skills=l.get_skills(),
            preferred_topics=l.get_preferred_topics()
        )
        for l in learners
    ]

@router.get("/{id}", response_model=LearnerProfileResponse)
def get_learner(id: str, db: Session = Depends(get_db)):
    """Fetch a single learner's detailed profile, skill mastery, and history."""
    profile = ProfileService.get_learner_profile_details(db, id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Learner with ID '{id}' not found.")
    return LearnerProfileResponse(**profile)

@router.post("/", response_model=LearnerResponse)
def create_learner(payload: LearnerCreate, db: Session = Depends(get_db)):
    """Register a new learner profile."""
    existing = ProfileService.get_learner(db, payload.learner_id)
    if existing:
        raise HTTPException(status_code=400, detail=f"Learner ID '{payload.learner_id}' already exists.")
    
    learner = ProfileService.create_learner(
        db=db,
        learner_id=payload.learner_id,
        name=payload.name,
        interests=payload.interests,
        education_background=payload.education_background,
        skills=payload.skills,
        preferred_topics=payload.preferred_topics
    )
    return LearnerResponse(
        learner_id=learner.learner_id,
        name=learner.name,
        interests=learner.get_interests(),
        education_background=learner.education_background,
        skills=learner.get_skills(),
        preferred_topics=learner.get_preferred_topics()
    )
