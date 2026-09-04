import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.database import init_db

try:
    from api.routes import (
        learners_router,
        courses_router,
        recommendations_router,
        feedback_router,
        evaluation_router
    )
except ImportError:
    from routes import (
        learners_router,
        courses_router,
        recommendations_router,
        feedback_router,
        evaluation_router
    )


app = FastAPI(
    title="Personalized Learning Recommendation with Explainability API",
    description="Hybrid Recommender System (Content-Based + Collaborative Filtering + Performance Scoring) with Explainable AI (XAI).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local Streamlit / web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route handlers
app.include_router(learners_router)
app.include_router(courses_router)
app.include_router(recommendations_router)
app.include_router(feedback_router)
app.include_router(evaluation_router)

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def root():
    return {
        "system": "Personalized Learning Recommendation System with Explainability",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/learners",
            "/learners/{id}",
            "/courses",
            "/recommendations/{learner_id}",
            "/feedback",
            "/evaluation/metrics"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

