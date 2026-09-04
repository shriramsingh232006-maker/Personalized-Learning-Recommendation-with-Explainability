import requests
from typing import Optional, Dict, Any

class OllamaClient:
    """
    Client for optional local phrasing of explanations using local Ollama (e.g., Phi-4-mini).
    Falls back safely if Ollama is not running.
    """
    def __init__(self, endpoint: str = "http://localhost:11434/api/generate", model: str = "phi4-mini"):
        self.endpoint = endpoint
        self.model = model

    def is_available(self) -> bool:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=1.0)
            return resp.status_code == 200
        except Exception:
            return False

    def polish_explanation(self, base_explanation: str, course_name: str, learner_name: str) -> str:
        """
        Polishes a template explanation into natural conversational phrasing.
        """
        if not self.is_available():
            return base_explanation

        prompt = (
            f"You are an academic learning assistant. Rephrase the following course recommendation "
            f"explanation nicely for learner {learner_name} about course '{course_name}'. "
            f"Keep it to 1-2 concise, motivating sentences.\n"
            f"Original: {base_explanation}\n"
            f"Rephrased:"
        )

        try:
            res = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=4.0
            )
            if res.status_code == 200:
                data = res.json()
                text = data.get("response", "").strip()
                return text if text else base_explanation
        except Exception:
            pass

        return base_explanation
