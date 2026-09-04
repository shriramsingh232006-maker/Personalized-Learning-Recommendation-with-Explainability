# Personalized Learning Recommendation System with Explainability (Hybrid Recommender + XAI)

A modular, production-ready Personalized Learning Recommender System integrating **Content-Based Filtering (CB)**, **Collaborative Filtering (CF)**, and **Skill/Performance Alignment Scoring (PS)** with **Explainable AI (XAI)** using rule-based reasoning, SHAP, and LIME feature attributions.

---

## 🌟 Key Features

1. **Hybrid Recommendation Engine ($\alpha=0.4, \beta=0.4, \gamma=0.2$)**:
   - **Content-Based Filtering**: TF-IDF semantic vectorization over course descriptions, tags, and skills with cosine similarity matching against learner interest profiles.
   - **Collaborative Filtering**: Truncated SVD matrix factorization on implicit/explicit learner-course interactions (ratings, quiz scores, completions).
   - **Performance Alignment**: Gaussian difficulty-vs-proficiency calibration scoring based on topic quiz mastery and prerequisite completion.
   - **Adaptive Cold-Start**: Automatically switches weights ($\alpha=0.7, \beta=0.0, \gamma=0.3$) for learners with fewer than 3 interactions.

2. **Explainable AI (XAI)**:
   - **Rule-Based Explainer (MVP Default)**: Transparent natural language explanations detailing exact factors (e.g., matched interests, prerequisite satisfaction, and quiz level).
   - **SHAP Feature Attribution**: Model-based surrogate explainer quantifying each component's contribution.
   - **LIME Local Importance**: Local feature sensitivity analysis.

3. **Full-Stack Implementation**:
   - **FastAPI Backend**: Async REST API with interactive Swagger documentation (`/docs`).
   - **Streamlit Interactive UI**: Real-time learner dashboard, recommendation cards, XAI breakdown charts, and feedback submission.
   - **SQLAlchemy + SQLite/PostgreSQL**: Robust data persistence for learners, courses, interactions, and skill profiles.
   - **Evaluation Module**: Offline computation of Precision@K, Recall@K, NDCG@K, and MAP.

---

## 📁 Directory Structure

```
personalized-learning-recommender/
|-- README.md
|-- requirements.txt
|-- config.yaml                 # Recommender weights, top-K, paths
|-- .env.example                # Environment variable configuration
|-- .gitignore
|
|-- data/
|   |-- raw/                    # Raw external datasets (OULAD, Kaggle, etc.)
|   |-- processed/              # Cleaned CSVs (learners, courses, interactions)
|   `-- synthetic/              # Synthetic generator & SQLite seeder
|       `-- generate_synthetic_data.py
|
|-- db/
|   |-- schema.sql              # Relational SQL schema
|   |-- database.py             # SQLAlchemy engine & session setup
|   `-- models.py               # SQLAlchemy ORM models
|
|-- src/
|   |-- profile/
|   |   `-- profile_service.py  # Learner Profile Module (LP)
|   |-- catalog/
|   |   `-- catalog_service.py  # Course Catalog Module (CM)
|   |-- recommenders/
|   |   |-- content_based.py    # TF-IDF + Cosine similarity (CB)
|   |   |-- collaborative.py    # SVD Matrix Factorization (CFI)
|   |   |-- performance_score.py # Skill-vs-Difficulty scoring
|   |   `-- hybrid_engine.py    # Weighted fusion & ranking (HE)
|   |-- explainability/
|   |   |-- rule_based_explainer.py # MVP default natural language XAI
|   |   |-- shap_explainer.py   # SHAP feature attribution
|   |   `-- lime_explainer.py   # LIME local feature sensitivity
|   |-- feedback/
|   |   `-- feedback_service.py # Feedback & topic skill profile updater (FB)
|   |-- evaluation/
|   |   `-- metrics.py          # Precision@K, Recall@K, NDCG@K, MAP (EV)
|   `-- llm/
|       `-- ollama_client.py    # Optional local LLM client for phrasing
|
|-- api/
|   |-- main.py                 # FastAPI application entrypoint
|   |-- routes/                 # Endpoint routers (learners, courses, recommendations, feedback, evaluation)
|   `-- schemas/                # Pydantic request & response models
|
|-- ui/
|   `-- streamlit_app.py        # Streamlit web dashboard
|
|-- models/                     # Pickled model artifacts (.pkl)
|-- notebooks/                  # Jupyter experiment notebooks (01_eda to 04_evaluation)
|-- tests/                      # Pytest unit & integration tests
`-- docs/                       # PRD, FRD, TRD documentation
```

---

## 🚀 Quick Start Guide

### 1. Set Up Virtual Environment & Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset & Seed Database

```bash
python data/synthetic/generate_synthetic_data.py
```
This will:
- Generate 25 curated courses across 6 technical domains.
- Generate 40 learners with distinct education, skills, and topic interests.
- Generate realistic interaction logs with ratings, quiz scores, and completions.
- Populate `learning_recommender.db` and save CSVs into `data/processed/`.

### 3. Run FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```
Interactive API docs will be live at: **http://127.0.0.1:8000/docs**

### 4. Run Streamlit Frontend

```bash
streamlit run ui/streamlit_app.py
```
The web dashboard opens at: **http://localhost:8501**

---

## 🧪 Running Automated Tests

Run the full Pytest suite:

```bash
pytest tests/ -v
```

---

## 📡 REST API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/learners` | List all registered learners |
| `GET` | `/learners/{id}` | Get detailed profile, topic skill mastery, and history |
| `POST` | `/learners` | Register a new learner profile |
| `GET` | `/courses` | List & filter courses by topic, difficulty, or search term |
| `POST` | `/courses` | Add or update a course in the catalog |
| `GET` | `/recommendations/{learner_id}?k=10` | Get Top-K personalized recommendations with XAI |
| `POST` | `/feedback` | Submit rating, quiz score, or completion status |
| `GET` | `/evaluation/metrics?k=5` | Retrieve offline Precision@K, Recall@K, NDCG@K, MAP |

---

## 📐 Hybrid Scoring Formula

$$\text{Hybrid\_Score} = \alpha \cdot \text{Content\_Score} + \beta \cdot \text{Collaborative\_Score} + \gamma \cdot \text{Performance\_Score}$$

- **Content-Score ($\alpha$)**: Cosine similarity between learner preferences + history and course TF-IDF vector.
- **Collaborative-Score ($\beta$)**: SVD rating prediction normalized to $[0, 1]$.
- **Performance-Score ($\gamma$)**: Gaussian alignment between learner's current topic skill and course difficulty level, plus prerequisite bonus.
