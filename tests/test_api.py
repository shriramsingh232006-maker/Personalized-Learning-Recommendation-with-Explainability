import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Personalized Learning Recommendation" in data["system"]

def test_get_courses():
    response = client.get("/courses")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
