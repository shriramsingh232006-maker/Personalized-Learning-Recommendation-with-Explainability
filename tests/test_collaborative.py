import pytest
from src.recommenders.collaborative import CollaborativeFilteringRecommender
from db.models import Course, Interaction

def test_collaborative_cold_start():
    recommender = CollaborativeFilteringRecommender(min_interactions=3)
    courses = [
        Course(course_id="C1", name="Course 1", topic="Topic", description="", difficulty="Beginner"),
        Course(course_id="C2", name="Course 2", topic="Topic", description="", difficulty="Beginner")
    ]
    interactions = [
        Interaction(learner_id="L1", course_id="C1", rating=5.0, completed=True)
    ]

    scores = recommender.score("L1", courses, interactions)
    # L1 has only 1 interaction, min_interactions is 3, so cold-start baseline is returned
    assert scores["C1"] == 0.5
    assert scores["C2"] == 0.5
