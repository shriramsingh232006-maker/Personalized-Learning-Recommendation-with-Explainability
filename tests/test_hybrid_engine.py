import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.database import Base
from db.models import Learner, Course, Interaction, SkillProfile
from src.recommenders.hybrid_engine import HybridEngine

@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Seed minimal data
    c1 = Course(
        course_id="C101",
        name="Intro Python",
        topic="Programming",
        description="Beginner python course",
        difficulty="Beginner",
        prerequisites="[]",
        skills_covered='["Python"]',
        tags='["python", "basics"]'
    )
    c2 = Course(
        course_id="C201",
        name="Advanced Python",
        topic="Programming",
        description="Metaprogramming and concurrency",
        difficulty="Advanced",
        prerequisites='["C101"]',
        skills_covered='["Python"]',
        tags='["python", "advanced"]'
    )
    db.add_all([c1, c2])

    l1 = Learner(
        learner_id="L001",
        name="Alice",
        interests='["Python"]',
        skills='["Python"]',
        preferred_topics='["Programming"]'
    )
    db.add(l1)
    db.commit()

    yield db
    db.close()

def test_hybrid_engine_recommendation(memory_db):
    engine = HybridEngine(alpha=0.5, beta=0.0, gamma=0.5)
    recs = engine.recommend(db=memory_db, learner_id="L001", k=2, filter_completed=False)

    assert len(recs) == 2
    assert recs[0]["hybrid_score"] >= recs[1]["hybrid_score"]
    assert "explanation_text" in recs[0]
    assert "factor_contributions" in recs[0]
