import pytest
import json
from src.recommenders.content_based import ContentBasedRecommender
from db.models import Learner, Course, Interaction

def test_content_based_scoring():
    courses = [
        Course(
            course_id="C1",
            name="Python for Data Science",
            topic="Data Science",
            description="Learn data manipulation with Python and Pandas.",
            difficulty="Beginner",
            prerequisites="[]",
            skills_covered='["Python", "Pandas"]',
            duration=10.0,
            tags='["python", "pandas", "data"]'
        ),
        Course(
            course_id="C2",
            name="Flutter Mobile Apps",
            topic="Mobile Development",
            description="Build iOS and Android apps with Dart and Flutter.",
            difficulty="Beginner",
            prerequisites="[]",
            skills_covered='["Flutter", "Dart"]',
            duration=15.0,
            tags='["flutter", "mobile", "dart"]'
        )
    ]

    learner = Learner(
        learner_id="L1",
        name="Test Learner",
        interests='["Data Science", "Python"]',
        skills='["Python"]',
        preferred_topics='["Data Science"]'
    )

    recommender = ContentBasedRecommender()
    recommender.fit(courses)
    scores = recommender.score(learner, courses, [])

    assert "C1" in scores
    assert "C2" in scores
    # C1 (Python/Data Science) should score significantly higher than C2 (Flutter) for this learner
    assert scores["C1"] > scores["C2"]
    assert 0.0 <= scores["C1"] <= 1.0
