from src.explainability.rule_based_explainer import RuleBasedExplainer
from src.explainability.shap_explainer import ShapExplainer
from src.explainability.lime_explainer import LimeExplainer
from db.models import Learner, Course

def test_rule_based_explainer():
    learner = Learner(
        learner_id="L1",
        name="Bob",
        interests='["Machine Learning"]',
        skills='["Python"]',
        preferred_topics='["Machine Learning"]'
    )
    course = Course(
        course_id="C1",
        name="Deep Learning",
        topic="Machine Learning",
        description="Deep neural networks",
        difficulty="Intermediate",
        prerequisites='["ML101"]',
        skills_covered='["Python", "PyTorch"]',
        tags='["machine learning", "deep learning"]'
    )

    scores = {
        "content_score": 0.85,
        "collaborative_score": 0.60,
        "performance_score": 0.70,
        "weight_content": 0.4,
        "weight_collaborative": 0.4,
        "weight_performance": 0.2
    }

    explanation = RuleBasedExplainer.explain(learner, course, scores, [], [])
    assert "explanation_text" in explanation
    assert len(explanation["explanation_text"]) > 10
    assert "primary_factor" in explanation
    assert len(explanation["factor_contributions"]) == 3

def test_shap_explainer():
    explainer = ShapExplainer()
    res = explainer.explain_instance(0.8, 0.6, 0.4)
    assert "shap_values" in res
    assert "Content_Score" in res["shap_values"]
    assert "prediction" in res

def test_lime_explainer():
    explainer = LimeExplainer()
    res = explainer.explain_instance(0.7, 0.5, 0.8)
    assert "local_importance" in res
    assert len(res["local_importance"]) == 3
