import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if not __package__:
    from api.routes.learners import router as learners_router
    from api.routes.courses import router as courses_router
    from api.routes.recommendations import router as recommendations_router
    from api.routes.feedback import router as feedback_router
    from api.routes.evaluation import router as evaluation_router
else:
    from .learners import router as learners_router
    from .courses import router as courses_router
    from .recommendations import router as recommendations_router
    from .feedback import router as feedback_router
    from .evaluation import router as evaluation_router

__all__ = [
    "learners_router",
    "courses_router",
    "recommendations_router",
    "feedback_router",
    "evaluation_router"
]

