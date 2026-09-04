import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if not __package__:
    from api.schemas.learner import LearnerCreate, LearnerResponse, LearnerProfileResponse
    from api.schemas.course import CourseCreate, CourseResponse
    from api.schemas.recommendation import RecommendationItem, RecommendationResponse, FeedbackCreate, FeedbackResponse, EvaluationMetricsResponse
else:
    from .learner import LearnerCreate, LearnerResponse, LearnerProfileResponse
    from .course import CourseCreate, CourseResponse
    from .recommendation import RecommendationItem, RecommendationResponse, FeedbackCreate, FeedbackResponse, EvaluationMetricsResponse

__all__ = [
    "LearnerCreate",
    "LearnerResponse",
    "LearnerProfileResponse",
    "CourseCreate",
    "CourseResponse",
    "RecommendationItem",
    "RecommendationResponse",
    "FeedbackCreate",
    "FeedbackResponse",
    "EvaluationMetricsResponse"
]

