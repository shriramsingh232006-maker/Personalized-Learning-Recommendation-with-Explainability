-- Schema for Personalized Learning Recommendation with Explainability

CREATE TABLE IF NOT EXISTS learners (
    learner_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    interests TEXT NOT NULL, -- JSON array of strings
    education_background TEXT,
    skills TEXT NOT NULL, -- JSON array of strings
    preferred_topics TEXT NOT NULL -- JSON array of strings
);

CREATE TABLE IF NOT EXISTS courses (
    course_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    topic TEXT NOT NULL,
    description TEXT NOT NULL,
    difficulty TEXT NOT NULL, -- Beginner, Intermediate, Advanced
    prerequisites TEXT, -- JSON array of strings or course_ids
    skills_covered TEXT NOT NULL, -- JSON array of strings
    duration REAL DEFAULT 10.0, -- hours
    tags TEXT NOT NULL -- JSON array of strings
);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    rating REAL DEFAULT 0.0, -- 1.0 to 5.0
    quiz_score REAL DEFAULT 0.0, -- 0.0 to 100.0
    completed INTEGER DEFAULT 0, -- 0 or 1
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS skill_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    skill_level REAL DEFAULT 0.5, -- 0.0 to 1.0
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE,
    UNIQUE(learner_id, topic)
);

CREATE TABLE IF NOT EXISTS recommendation_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    hybrid_score REAL NOT NULL,
    explanation_text TEXT NOT NULL,
    generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (learner_id) REFERENCES learners(learner_id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);
