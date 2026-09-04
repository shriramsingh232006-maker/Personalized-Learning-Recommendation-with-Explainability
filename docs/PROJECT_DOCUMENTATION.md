# Personalized Learning Recommendation System with Explainability (Hybrid Recommender + XAI)
## Comprehensive Technical & Architecture Overview

---

## 1. Executive Summary & Problem Statement

Modern e-learning platforms often face two primary challenges:
1. **The Choice Overload & Cold-Start Problem**: Learners are overwhelmed by thousands of courses and do not know where to begin, while new users lack sufficient historical interaction data for standard collaborative recommenders.
2. **The "Black-Box" Recommender Dilemma**: Deep or complex recommendation models often output suggestions without explaining *why* a particular course was chosen, which degrades student trust and motivation.

This project implements an end-to-end, production-grade **Personalized Learning Recommendation System with Explainable AI (XAI)**. It uniquely unifies:
- **Content-Based Filtering (CB)** based on semantic textual relevance.
- **Collaborative Filtering (CF)** using latent factor matrix factorization (SVD).
- **Skill-vs-Difficulty Calibration (Performance Score)** ensuring courses match the learner's actual mastery and prerequisite background.
- **Multi-tiered Explainable AI (XAI)** providing human-readable justifications backed by **Rule-Based reasoning**, **SHAP feature attribution**, and **LIME local sensitivity**.

---

## 2. High-Level System Architecture

The project follows a modular, layered architecture:

```
[ Streamlit Web Dashboard ] (Port 8501)
         |
         | HTTP / Python Service Calls
         v
[ FastAPI REST API Backend ] (Port 8000)
         |
         +--> Learner Profile Service (LP)
         +--> Course Catalog Service (CM)
         +--> Hybrid Recommender Engine (HE)
         |        |-- Content-Based Filtering (CB) [TF-IDF + Cosine Similarity]
         |        |-- Collaborative Filtering (CFI) [Truncated SVD Matrix Factorization]
         |        |-- Performance Scorer (PS) [Skill-vs-Difficulty Gaussian Calibration]
         |        `-- Explainability Module (XM) [Rule-Based + SHAP + LIME]
         +--> Feedback & Topic Skill Updater (FB)
         `--> Evaluation Engine (EV) [Precision@K, Recall@K, NDCG@K, MAP]
         |
         v
[ SQLite Database: learning_recommender.db ] + [ Serialized Models: models/*.pkl ]
```

---

## 3. Recommender Algorithms & Mathematical Formulations

### 3.1 Content-Based Filtering Module (CB)
* **Goal**: Measure how well course content aligns with a learner's stated interests, current skills, and history.
* **Mechanism**:
  1. Each course $c$ is represented as a text document combining its title, topic, description, difficulty, prerequisites, and tags.
  2. TF-IDF vectorization generates a sparse feature representation $\mathbf{v}_c \in \mathbb{R}^{d}$.
  3. The learner's profile vector $\mathbf{u}_l$ is formed as a weighted blend of their stated interests ($40\%$) and the average TF-IDF vectors of their completed/highly-rated courses ($60\%$):
     $$\mathbf{u}_l = 0.4 \cdot \mathbf{v}_{\text{profile}} + 0.6 \cdot \frac{1}{\sum w_i} \sum_{i \in \text{history}} w_i \mathbf{v}_{c_i}$$
  4. Content score is computed using normalized Cosine Similarity:
     $$\text{Content\_Score}(l, c) = \frac{\mathbf{u}_l \cdot \mathbf{v}_c}{\|\mathbf{u}_l\|_2 \|\mathbf{v}_c\|_2} \in [0.0, 1.0]$$

---

### 3.2 Collaborative Filtering Module (CF)
* **Goal**: Predict a learner's preference based on collective behavior of similar peers.
* **Mechanism**:
  1. An interaction matrix $\mathbf{R} \in \mathbb{R}^{M \times N}$ is constructed from user interactions, combining ratings (1–5), quiz scores, and completion bonuses:
     $$R_{l, c} = \text{rating} + \mathbb{I}(\text{completed}) \cdot 0.5$$
  2. **Truncated SVD (Singular Value Decomposition)** factorizes the matrix into latent factor representations:
     $$\mathbf{R} \approx \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
  3. Predicted rating $\hat{R}_{l, c}$ is normalized to a $[0.0, 1.0]$ scale:
     $$\text{Collaborative\_Score}(l, c) = \frac{\hat{R}_{l, c} - 1.0}{4.0}$$
  4. **Cold-Start Rule**: If a learner has fewer than 3 interactions ($|\text{Interactions}| < 3$), collaborative filtering is unreliable. The engine dynamically sets $\beta = 0.0$ and reallocates weight to Content and Skill alignment.

---

### 3.3 Skill & Performance Alignment Scoring (PS)
* **Goal**: Ensure the recommended course is neither discouragingly hard nor trivially easy.
* **Mechanism**:
  1. Each course difficulty level is mapped to a target skill mastery level:
     - Beginner: 0.30
     - Intermediate: 0.65
     - Advanced: 0.90
  2. The learner's topic mastery $S_{l, \text{topic}}$ is obtained from their `SkillProfile` (calibrated from past quiz scores).
  3. A Gaussian alignment penalty evaluates the difference:
     $$\text{Alignment}(l, c) = \exp\left(-1.5 \cdot (S_{l, \text{topic}} - \text{Target\_Skill})^2\right)$$
  4. Prerequisite verification checks whether the learner has completed prerequisite courses:
     $$\text{Prereq\_Score}(l, c) = \frac{\sum_{p \in \text{Prereqs}} \mathbb{I}(p \in \text{Completed})}{\max(1, |\text{Prereqs}|)}$$
  5. Final performance score:
     $$\text{Performance\_Score}(l, c) = 0.75 \cdot \text{Alignment} + 0.25 \cdot \text{Prereq\_Score} \in [0.0, 1.0]$$

---

### 3.4 Hybrid Fusion
The overall match score is a weighted linear fusion configured via `config.yaml`:

$$\text{Hybrid\_Score} = \alpha \cdot \text{Content\_Score} + \beta \cdot \text{Collaborative\_Score} + \gamma \cdot \text{Performance\_Score}$$

* **Default Weights**: $\alpha = 0.4$, $\beta = 0.4$, $\gamma = 0.2$ ($\alpha + \beta + \gamma = 1.0$).
* **Cold-Start Weights**: $\alpha = 0.7$, $\beta = 0.0$, $\gamma = 0.3$.
* Top-$K$ highest-scoring courses that the user hasn't already completed are ranked and returned.

---

## 4. Explainable AI (XAI) Architecture

Explainability is a first-class citizen of this architecture, offering three complementary explanation methods:

1. **Rule-Based Explainer (MVP Default)**:
   - Identifies the highest weighted scoring component (e.g. Content Relevance vs Peer Trends vs Skill Alignment).
   - Generates natural, human-understandable evidence sentences referencing specific completed prerequisites, matched tags, and quiz mastery.
2. **SHAP Feature Attribution**:
   - Fits a surrogate regression model over scoring features and computes Shapley values quantifying how each dimension shifts the score above baseline.
3. **LIME Explainer**:
   - Performs localized perturbation sampling around the instance to estimate local sensitivity gradients.
4. **Optional LLM Phrasing (`ollama_client.py`)**:
   - Supports local models (such as Phi-4-mini) through an offline Ollama endpoint for conversational rephrasing without sending student data to third-party APIs.

---

## 5. Database Schema & Data Modeling

The database uses SQLite via SQLAlchemy ORM with foreign key integrity:

| Table | Key Columns | Purpose |
|---|---|---|
| `learners` | `learner_id` (PK), `name`, `interests` (JSON), `education_background`, `skills` (JSON), `preferred_topics` (JSON) | Learner profile and preferences |
| `courses` | `course_id` (PK), `name`, `topic`, `description`, `difficulty`, `prerequisites` (JSON), `skills_covered` (JSON), `duration`, `tags` (JSON) | Course catalog |
| `interactions` | `interaction_id` (PK), `learner_id` (FK), `course_id` (FK), `rating`, `quiz_score`, `completed` (bool), `timestamp` | Learner activity and performance |
| `skill_profiles` | `id` (PK), `learner_id` (FK), `topic`, `skill_level` ($0.0 - 1.0$), `last_updated` | Dynamic topic proficiency |
| `recommendation_logs` | `log_id` (PK), `learner_id` (FK), `course_id` (FK), `hybrid_score`, `explanation_text`, `generated_at` | Audit trail and XAI log |

---

## 6. REST API Design (FastAPI)

All endpoints return JSON and are documented interactively at `/docs`:

```
GET  /learners                  List all learners
GET  /learners/{id}             Get learner profile, history, and topic skill mastery
POST /learners                  Register a new learner
GET  /courses                   Search and filter courses (query: topic, difficulty, search)
GET  /courses/{id}              Get course details
POST /courses                   Add/update a course in the catalog
GET  /recommendations/{id}      Generate Top-K hybrid recommendations with XAI
POST /feedback                  Submit rating, quiz score, or completion (updates skill profile)
GET  /evaluation/metrics        Compute offline Precision@K, Recall@K, NDCG@K, and MAP
```

---

## 7. Interactive Streamlit Dashboard (`ui/streamlit_app.py`)

The user interface provides 4 specialized tabs:
1. **🎯 Recommendations & XAI**:
   - Course cards displaying duration, difficulty badges, and match percentage.
   - Highlighted natural language **Why Recommended** box.
   - Interactive Plotly horizontal bar charts for **Component Breakdown** and **SHAP Attribution**.
   - Inline feedback widget allowing learners to rate courses, record quiz scores, and instantly observe profile updates.
2. **👤 Learner Profile**:
   - Summary of completed courses, average ratings, and quiz performance.
   - Plotly Radar chart depicting the learner's multi-topic mastery levels.
3. **📚 Course Catalog**:
   - Search by keyword, filter by topic, and inspect prerequisites and syllabus tags.
4. **📊 System Evaluation**:
   - Real-time offline benchmark testing computing Precision@K, Recall@K, NDCG@K, and MAP across interaction splits.

---

## 8. Offline Evaluation Metrics

To rigorously assess recommendation quality, the evaluation module (`src/evaluation/metrics.py`) computes:
- **Precision@K**: Ratio of relevant items found in the top-K recommendations.
- **Recall@K**: Ratio of all relevant items captured in the top-K recommendations.
- **NDCG@K**: Discounted ranking reward giving higher value to relevant items ranked at positions 1, 2, 3...
- **MAP@K**: Mean Average Precision over all evaluated learners.

---

## 9. Verification & Running Instructions

```powershell
# 1. Navigate to directory
cd "C:\Users\Ace\OneDrive\Desktop\Personalized Learning Recommendation with Explainability"

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Launch FastAPI Backend Server (Port 8000)
python -m uvicorn api.main:app --reload --port 8000

# 4. Launch Streamlit Frontend Dashboard (Port 8501)
streamlit run ui/streamlit_app.py

# 5. Run Pytest Suite
pytest tests/ -v
```
