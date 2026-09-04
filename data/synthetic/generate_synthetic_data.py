import os
import sys
import json
import random
import csv
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from db.database import SessionLocal, init_db
    from db.models import Learner, Course, Interaction, SkillProfile
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

random.seed(42)

TOPICS = [
    "Machine Learning",
    "Data Science",
    "Web Development",
    "Cloud Computing",
    "Cybersecurity",
    "Mobile Development"
]

SKILLS_BY_TOPIC = {
    "Machine Learning": ["Python", "PyTorch", "TensorFlow", "Scikit-Learn", "Deep Learning", "NLP", "Computer Vision"],
    "Data Science": ["Python", "Pandas", "SQL", "Data Visualization", "Statistics", "Exploratory Data Analysis"],
    "Web Development": ["HTML", "CSS", "JavaScript", "React", "Node.js", "FastAPI", "REST APIs"],
    "Cloud Computing": ["AWS", "Docker", "Kubernetes", "Google Cloud", "CI/CD", "Terraform", "Linux"],
    "Cybersecurity": ["Network Security", "Cryptography", "Penetration Testing", "Ethical Hacking", "Wireshark", "OWASP"],
    "Mobile Development": ["Flutter", "Dart", "Swift", "Kotlin", "React Native", "Android SDK", "iOS Development"]
}

COURSE_TEMPLATES = [
    # Machine Learning
    ("ML101", "Introduction to Machine Learning", "Machine Learning", "Learn core machine learning algorithms including linear regression, logistic regression, and decision trees with Scikit-learn.", "Beginner", [], ["Python", "Scikit-Learn"], 12.0, ["ml", "ai", "beginner", "python"]),
    ("ML201", "Deep Learning & Neural Networks", "Machine Learning", "Understand neural networks, backpropagation, CNNs, and RNNs using PyTorch and TensorFlow.", "Intermediate", ["ML101"], ["Python", "PyTorch", "Deep Learning"], 20.0, ["deep learning", "neural networks", "pytorch"]),
    ("ML301", "Natural Language Processing with Transformers", "Machine Learning", "Advanced NLP techniques covering word embeddings, attention mechanisms, BERT, and GPT architectures.", "Advanced", ["ML201"], ["NLP", "PyTorch", "Deep Learning"], 25.0, ["nlp", "transformers", "llm"]),
    ("ML302", "Computer Vision Systems", "Machine Learning", "Image classification, object detection, segmentation, and feature extraction using OpenCV and PyTorch.", "Advanced", ["ML201"], ["Computer Vision", "PyTorch"], 22.0, ["vision", "opencv", "pytorch"]),
    ("ML202", "Practical Machine Learning in Production", "Machine Learning", "Deploying and serving ML models using FastAPI, Docker, model monitoring, and MLflow.", "Intermediate", ["ML101"], ["Python", "Docker", "FastAPI"], 16.0, ["mlops", "deployment", "docker"]),

    # Data Science
    ("DS101", "Python for Data Analysis", "Data Science", "Master NumPy, Pandas, and Matplotlib for manipulating, cleaning, and visualizing structured datasets.", "Beginner", [], ["Python", "Pandas", "Data Visualization"], 10.0, ["python", "pandas", "data analysis"]),
    ("DS102", "SQL for Data Wrangling", "Data Science", "Learn relational database querying, joins, window functions, and data transformations with SQL.", "Beginner", [], ["SQL"], 8.0, ["sql", "databases", "queries"]),
    ("DS201", "Statistical Modeling & Hypothesis Testing", "Data Science", "Probability distributions, hypothesis testing, A/B testing, and regression analysis for data science.", "Intermediate", ["DS101"], ["Statistics", "Python"], 15.0, ["statistics", "ab testing", "probability"]),
    ("DS202", "Exploratory Data Analysis & Feature Engineering", "Data Science", "Advanced techniques for discovering patterns, handling missing values, and generating predictive features.", "Intermediate", ["DS101"], ["Pandas", "Exploratory Data Analysis"], 14.0, ["eda", "feature engineering", "pandas"]),
    ("DS301", "Big Data Analytics with PySpark", "Data Science", "Distributed computing principles, resilient distributed datasets (RDDs), and scalable analytics using PySpark.", "Advanced", ["DS101", "DS102"], ["Python", "SQL"], 24.0, ["spark", "big data", "distributed"]),

    # Web Development
    ("WD101", "Modern Web Development Fundamentals", "Web Development", "Building responsive, modern web applications from scratch with HTML5, CSS3, and modern JavaScript.", "Beginner", [], ["HTML", "CSS", "JavaScript"], 10.0, ["frontend", "html", "css", "javascript"]),
    ("WD201", "Interactive Frontend Development with React", "Web Development", "Component architecture, hooks, state management, and building dynamic single-page applications.", "Intermediate", ["WD101"], ["JavaScript", "React"], 18.0, ["react", "frontend", "spa"]),
    ("WD202", "RESTful API Engineering with FastAPI", "Web Development", "Building high-performance, asynchronous REST APIs with Python, FastAPI, Pydantic, and SQLite/PostgreSQL.", "Intermediate", [], ["Python", "FastAPI", "REST APIs"], 14.0, ["backend", "fastapi", "python", "api"]),
    ("WD203", "Full-Stack Node.js & Express", "Web Development", "Server-side JavaScript programming, routing, middleware, database integration, and authentication.", "Intermediate", ["WD101"], ["JavaScript", "Node.js", "REST APIs"], 16.0, ["backend", "nodejs", "javascript"]),
    ("WD301", "Scalable Microservices Architecture", "Web Development", "Architecting resilient microservices with event-driven design, message queues, and API gateways.", "Advanced", ["WD202"], ["FastAPI", "Docker", "REST APIs"], 22.0, ["microservices", "architecture", "scalability"]),

    # Cloud Computing
    ("CC101", "Cloud Computing & AWS Essentials", "Cloud Computing", "Foundational cloud concepts, AWS EC2, S3, IAM, VPC, and serverless architectures.", "Beginner", [], ["AWS", "Linux"], 12.0, ["aws", "cloud", "serverless"]),
    ("CC201", "Containerization with Docker & Kubernetes", "Cloud Computing", "Building container images, managing multi-container deployments with Docker Compose, and Kubernetes orchestration.", "Intermediate", ["CC101"], ["Docker", "Kubernetes", "Linux"], 18.0, ["docker", "kubernetes", "containers"]),
    ("CC202", "Infrastructure as Code with Terraform", "Cloud Computing", "Automating multi-cloud infrastructure provisioning and state management using declarative Terraform HCL.", "Intermediate", ["CC101"], ["Terraform", "AWS"], 15.0, ["iac", "terraform", "devops"]),
    ("CC301", "Enterprise CI/CD Pipelines & DevOps", "Cloud Computing", "Continuous integration and automated delivery pipelines with GitHub Actions, testing, and cloud deployment.", "Advanced", ["CC201"], ["CI/CD", "Docker", "Linux"], 20.0, ["cicd", "devops", "automation"]),

    # Cybersecurity
    ("CS101", "Cybersecurity Principles & Threat Landscape", "Cybersecurity", "Fundamental principles of information security, CIA triad, threat vectors, authentication, and hygiene.", "Beginner", [], ["Network Security"], 10.0, ["security", "infosec", "beginner"]),
    ("CS201", "Applied Cryptography & Secure Communications", "Cybersecurity", "Symmetric and asymmetric encryption, hashing algorithms, digital certificates, and TLS protocols.", "Intermediate", ["CS101"], ["Cryptography", "Network Security"], 16.0, ["crypto", "encryption", "tls"]),
    ("CS202", "Web Application Penetration Testing", "Cybersecurity", "Identifying and exploiting web security vulnerabilities: SQL injection, XSS, CSRF, and OWASP Top 10.", "Intermediate", ["CS101"], ["OWASP", "Penetration Testing"], 18.0, ["penetration testing", "owasp", "ethical hacking"]),
    ("CS301", "Network Forensics & Packet Analysis", "Cybersecurity", "Inspecting malicious network traffic, protocol anomalies, intrusion detection, and Wireshark packet analysis.", "Advanced", ["CS101"], ["Network Security", "Wireshark"], 22.0, ["forensics", "wireshark", "packets"]),

    # Mobile Development
    ("MD101", "Cross-Platform Mobile Apps with Flutter", "Mobile Development", "Building responsive, modern iOS and Android apps using Google Flutter framework and Dart.", "Beginner", [], ["Flutter", "Dart"], 16.0, ["flutter", "dart", "mobile"]),
    ("MD201", "State Management in Flutter & Dart", "Mobile Development", "Architecting robust Flutter apps with Provider, Riverpod, and BLoC state management patterns.", "Intermediate", ["MD101"], ["Flutter", "Dart"], 15.0, ["flutter", "bloc", "state management"]),
    ("MD202", "Native Android App Development with Kotlin", "Mobile Development", "Modern Android UI with Jetpack Compose, Room database, Coroutines, and MVVM architecture.", "Intermediate", [], ["Kotlin", "Android SDK"], 20.0, ["android", "kotlin", "mobile"]),
    ("MD301", "Mobile App Performance & Offline Sync", "Mobile Development", "Optimizing memory, frame rates, background processing, and local database sync for enterprise mobile apps.", "Advanced", ["MD101"], ["Flutter", "Kotlin"], 18.0, ["performance", "mobile", "sync"])
]

NAMES = [
    "Aarav Sharma", "Ananya Verma", "Rohan Mehta", "Priya Nair", "Aditya Patel",
    "Neha Gupta", "Vikram Rao", "Sneha Joshi", "Kabir Singh", "Divya Iyer",
    "Arjun Kapoor", "Ishita Roy", "Karan Reddy", "Tanvi Bhat", "Rahul Deshmukh",
    "Pooja Das", "Manish Malhotra", "Kavya Menon", "Siddharth Jain", "Ritu Agrawal"
]

DEGREES = [
    "B.Tech Computer Science",
    "B.Sc Information Technology",
    "B.E. Electronics & Communication",
    "Master of Computer Applications",
    "B.Sc Data Science",
    "Self-Taught Developer"
]

def generate_data(num_learners=40, min_interactions=5, max_interactions=12):
    courses_data = []
    for cid, name, topic, desc, diff, prereq, skills, dur, tags in COURSE_TEMPLATES:
        courses_data.append({
            "course_id": cid,
            "name": name,
            "topic": topic,
            "description": desc,
            "difficulty": diff,
            "prerequisites": json.dumps(prereq),
            "skills_covered": json.dumps(skills),
            "duration": dur,
            "tags": json.dumps(tags)
        })

    learners_data = []
    for i in range(1, num_learners + 1):
        lid = f"L{i:03d}"
        name = NAMES[(i - 1) % len(NAMES)] if i <= len(NAMES) else f"Learner {i}"
        
        # Pick 1-2 primary preferred topics
        pref_topics = random.sample(TOPICS, k=random.randint(1, 2))
        
        # Select interests based on preferred topics
        interests = set()
        skills = set()
        for top in pref_topics:
            skills.update(random.sample(SKILLS_BY_TOPIC[top], k=min(3, len(SKILLS_BY_TOPIC[top]))))
            interests.add(top)
            interests.add(f"{top} Projects")

        learners_data.append({
            "learner_id": lid,
            "name": name,
            "interests": json.dumps(list(interests)),
            "education_background": random.choice(DEGREES),
            "skills": json.dumps(list(skills)),
            "preferred_topics": json.dumps(pref_topics)
        })

    # Generate Interactions & Skill Profiles
    interactions_data = []
    skill_profiles_data = []
    
    for l in learners_data:
        lid = l["learner_id"]
        pref_topics = json.loads(l["preferred_topics"])
        
        # Determine learner's typical skill level per topic
        topic_skills = {}
        for top in TOPICS:
            if top in pref_topics:
                topic_skills[top] = round(random.uniform(0.6, 0.95), 2)
            else:
                topic_skills[top] = round(random.uniform(0.2, 0.5), 2)
            skill_profiles_data.append({
                "learner_id": lid,
                "topic": top,
                "skill_level": topic_skills[top]
            })

        # Generate interactions (biased toward preferred topics)
        num_int = random.randint(min_interactions, max_interactions)
        interacted_courses = set()

        for _ in range(num_int):
            if random.random() < 0.75:
                eligible = [c for c in courses_data if c["topic"] in pref_topics and c["course_id"] not in interacted_courses]
            else:
                eligible = [c for c in courses_data if c["course_id"] not in interacted_courses]

            if not eligible:
                continue

            course = random.choice(eligible)
            cid = course["course_id"]
            interacted_courses.add(cid)

            # Score depends on difficulty and learner skill in topic
            c_diff = course["difficulty"]
            l_skill = topic_skills[course["topic"]]

            # Determine quiz score
            diff_factor = 0.9 if c_diff == "Beginner" else (0.75 if c_diff == "Intermediate" else 0.6)
            base_score = (l_skill * 70) + (diff_factor * 30) + random.uniform(-10, 10)
            quiz_score = max(35.0, min(100.0, round(base_score, 1)))

            # Completed if quiz score >= 60
            completed = bool(quiz_score >= 60.0)
            
            # Rating correlates with quiz performance and completion
            rating_val = 3.0 + (quiz_score / 50.0) + random.uniform(-0.5, 0.5)
            rating = max(1.0, min(5.0, round(rating_val, 1)))

            days_ago = random.randint(1, 120)
            timestamp = datetime.utcnow() - timedelta(days=days_ago)

            interactions_data.append({
                "learner_id": lid,
                "course_id": cid,
                "rating": rating,
                "quiz_score": quiz_score,
                "completed": completed,
                "timestamp": timestamp
            })

    return courses_data, learners_data, interactions_data, skill_profiles_data

def _write_csv(filepath, data_list):
    if not data_list:
        return
    keys = data_list[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in data_list:
            # Format datetime to string if needed
            cleaned = {}
            for k, v in row.items():
                if isinstance(v, datetime):
                    cleaned[k] = v.isoformat()
                else:
                    cleaned[k] = v
            writer.writerow(cleaned)

def save_and_seed():
    print("Generating synthetic datasets...")
    courses_data, learners_data, interactions_data, skill_profiles_data = generate_data()

    # Save to processed CSVs
    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(processed_dir / "courses.csv", courses_data)
    _write_csv(processed_dir / "learners.csv", learners_data)
    _write_csv(processed_dir / "interactions.csv", interactions_data)
    _write_csv(processed_dir / "skill_profiles.csv", skill_profiles_data)

    print(f"Saved CSVs to {processed_dir}:")
    print(f"  - courses: {len(courses_data)}")
    print(f"  - learners: {len(learners_data)}")
    print(f"  - interactions: {len(interactions_data)}")
    print(f"  - skill profiles: {len(skill_profiles_data)}")

    # Seed Database
    print("Initializing and seeding SQLite database...")
    db_file = BASE_DIR / "learning_recommender.db"
    
    if HAS_SQLALCHEMY:
        init_db()
        db = SessionLocal()
        try:
            db.query(Interaction).delete()
            db.query(SkillProfile).delete()
            db.query(Course).delete()
            db.query(Learner).delete()
            db.commit()

            for c in courses_data:
                db.add(Course(**c))
            db.commit()

            for l in learners_data:
                db.add(Learner(**l))
            db.commit()

            for sp in skill_profiles_data:
                db.add(SkillProfile(**sp))
            db.commit()

            for it in interactions_data:
                db.add(Interaction(**it))
            db.commit()

            print("Database successfully initialized and seeded via SQLAlchemy!")
        except Exception as e:
            db.rollback()
            print(f"SQLAlchemy seeding error, falling back to direct SQLite: {e}")
            _seed_sqlite_direct(db_file, courses_data, learners_data, interactions_data, skill_profiles_data)
        finally:
            db.close()
    else:
        _seed_sqlite_direct(db_file, courses_data, learners_data, interactions_data, skill_profiles_data)

def _seed_sqlite_direct(db_file, courses_data, learners_data, interactions_data, skill_profiles_data):
    schema_path = BASE_DIR / "db" / "schema.sql"
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            cur.executescript(f.read())

    cur.execute("DELETE FROM interactions")
    cur.execute("DELETE FROM skill_profiles")
    cur.execute("DELETE FROM courses")
    cur.execute("DELETE FROM learners")

    for c in courses_data:
        cur.execute(
            "INSERT INTO courses (course_id, name, topic, description, difficulty, prerequisites, skills_covered, duration, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c["course_id"], c["name"], c["topic"], c["description"], c["difficulty"], c["prerequisites"], c["skills_covered"], c["duration"], c["tags"])
        )

    for l in learners_data:
        cur.execute(
            "INSERT INTO learners (learner_id, name, interests, education_background, skills, preferred_topics) VALUES (?, ?, ?, ?, ?, ?)",
            (l["learner_id"], l["name"], l["interests"], l["education_background"], l["skills"], l["preferred_topics"])
        )

    for sp in skill_profiles_data:
        cur.execute(
            "INSERT INTO skill_profiles (learner_id, topic, skill_level) VALUES (?, ?, ?)",
            (sp["learner_id"], sp["topic"], sp["skill_level"])
        )

    for it in interactions_data:
        cur.execute(
            "INSERT INTO interactions (learner_id, course_id, rating, quiz_score, completed, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (it["learner_id"], it["course_id"], it["rating"], it["quiz_score"], 1 if it["completed"] else 0, it["timestamp"].isoformat() if isinstance(it["timestamp"], datetime) else it["timestamp"])
        )

    conn.commit()
    conn.close()
    print("Database successfully initialized and seeded directly with sqlite3!")

if __name__ == "__main__":
    save_and_seed()
