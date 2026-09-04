from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import Ridge

class ShapExplainer:
    """
    Model-based Explainer using SHAP feature attribution over a surrogate model
    fitted to explain hybrid recommendation scores.
    """
    def __init__(self):
        self.feature_names = ["Content_Score", "Collaborative_Score", "Performance_Score"]
        self.surrogate_model = Ridge(alpha=1.0)
        self.is_fitted = False
        self._init_surrogate()

    def _init_surrogate(self):
        # Generate synthetic grid to fit the surrogate model
        rng = np.random.default_rng(42)
        X = rng.uniform(0.0, 1.0, size=(200, 3))
        # y = 0.4*x0 + 0.4*x1 + 0.2*x2
        y = 0.4 * X[:, 0] + 0.4 * X[:, 1] + 0.2 * X[:, 2] + rng.normal(0, 0.01, size=200)
        self.surrogate_model.fit(X, y)
        self.is_fitted = True

    def explain_instance(self, content_score: float, collaborative_score: float, performance_score: float) -> Dict[str, Any]:
        x = np.array([[content_score, collaborative_score, performance_score]])
        pred = float(self.surrogate_model.predict(x)[0])
        
        # Exact linear SHAP values: coef_ * (x - mean(X))
        base_value = float(self.surrogate_model.intercept_ + np.dot(self.surrogate_model.coef_, [0.5, 0.5, 0.5]))
        shap_values = self.surrogate_model.coef_ * (x[0] - 0.5)

        attributions = {}
        for name, val in zip(self.feature_names, shap_values):
            attributions[name] = round(float(val), 4)

        return {
            "base_value": round(base_value, 4),
            "prediction": round(pred, 4),
            "shap_values": attributions,
            "top_positive_feature": max(attributions, key=attributions.get)
        }
