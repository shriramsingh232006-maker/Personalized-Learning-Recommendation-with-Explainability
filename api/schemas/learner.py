from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class LearnerBase(BaseModel):
    learner_id: str
    name: str
    interests: List[str] = []
    education_background: Optional[str] = None
    skills: List[str] = []
    preferred_topics: List[str] = []

class LearnerCreate(LearnerBase):
    pass

class LearnerResponse(LearnerBase):
    class Config:
        from_attributes = True

class LearnerProfileResponse(BaseModel):
    learner_id: str
    name: str
    education_background: Optional[str]
    interests: List[str]
    skills: List[str]
    preferred_topics: List[str]
    completed_courses: List[Dict[str, Any]]
    in_progress_courses: List[Dict[str, Any]]
    skill_profile: Dict[str, float]
    average_quiz_score: float
    average_rating: float
    total_interactions: int
