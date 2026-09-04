from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class FactorContribution(BaseModel):
    name: str
    weighted_score: float
    raw_score: float

class RecommendationItem(BaseModel):
    course_id: str
    name: str
    topic: str
    difficulty: str
    duration: float
    hybrid_score: float
    content_score: float
    collaborative_score: float
    performance_score: float
    is_cold_start: bool
    explanation_text: str
    primary_factor: str
    factor_contributions: List[FactorContribution]
    shap_values: Dict[str, float]
    tags: List[str]
    skills_covered: List[str]

class RecommendationResponse(BaseModel):
    learner_id: str
    k: int
    recommendations: List[RecommendationItem]

class FeedbackCreate(BaseModel):
    learner_id: str
    course_id: str
    rating: Optional[float] = None
    quiz_score: Optional[float] = None
    completed: Optional[bool] = None

class FeedbackResponse(BaseModel):
    status: str
    interaction_id: int
    learner_id: str
    course_id: str
    rating: float
    quiz_score: float
    completed: bool
    topic: str
    updated_topic_skill_level: float

class EvaluationMetricsResponse(BaseModel):
    k: int
    total_evaluated_learners: int
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    map_at_k: float
