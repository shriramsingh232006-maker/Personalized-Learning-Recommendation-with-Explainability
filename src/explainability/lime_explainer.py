from typing import Dict, Any, List
import numpy as np

class LimeExplainer:
    """
    Local Interpretable Model-agnostic Explanations (LIME) style explainer.
    Perturbs local feature inputs to measure local sensitivity/impact on hybrid score.
    """
    def __init__(self):
        self.feature_names = ["Content Relevance", "Collaborative Filter", "Performance Alignment"]

    def explain_instance(
        self,
        content_score: float,
        collaborative_score: float,
        performance_score: float,
        weights: List[float] = [0.4, 0.4, 0.2],
        num_samples: int = 50
    ) -> Dict[str, Any]:
        center = np.array([content_score, collaborative_score, performance_score])
        w = np.array(weights)
        center_score = np.dot(center, w)

        # Generate local perturbations
        rng = np.random.default_rng(42)
        perturbations = rng.normal(0, 0.05, size=(num_samples, 3))
        perturbed_samples = np.clip(center + perturbations, 0.0, 1.0)
        perturbed_scores = np.dot(perturbed_samples, w)

        # Calculate local linear weights via weighted least squares
        distances = np.linalg.norm(perturbations, axis=1)
        kernel_weights = np.exp(-(distances ** 2) / (0.1 ** 2))

        # Local slope/impact
        impacts = {}
        for idx, feat_name in enumerate(self.feature_names):
            diff = perturbed_samples[:, idx] - center[idx]
            impact = float(np.sum(kernel_weights * diff * (perturbed_scores - center_score)) / (np.sum(kernel_weights * (diff ** 2)) + 1e-6))
            impacts[feat_name] = round(impact * center[idx], 4)

        return {
            "center_score": round(float(center_score), 4),
            "local_importance": impacts,
            "top_contributor": max(impacts, key=impacts.get)
        }
