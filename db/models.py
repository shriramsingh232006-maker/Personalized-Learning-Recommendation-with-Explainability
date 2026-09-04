import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from db.database import Base

class Learner(Base):
    __tablename__ = "learners"

    learner_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    interests = Column(Text, default="[]")  # JSON string
    education_background = Column(String(100), nullable=True)
    skills = Column(Text, default="[]")  # JSON string
    preferred_topics = Column(Text, default="[]")  # JSON string

    interactions = relationship("Interaction", back_populates="learner", cascade="all, delete-orphan")
    skill_profiles = relationship("SkillProfile", back_populates="learner", cascade="all, delete-orphan")
    recommendation_logs = relationship("RecommendationLog", back_populates="learner", cascade="all, delete-orphan")

    def get_interests(self):
        try:
            return json.loads(self.interests) if self.interests else []
        except Exception:
            return []

    def get_skills(self):
        try:
            return json.loads(self.skills) if self.skills else []
        except Exception:
            return []

    def get_preferred_topics(self):
        try:
            return json.loads(self.preferred_topics) if self.preferred_topics else []
        except Exception:
            return []


class Course(Base):
    __tablename__ = "courses"

    course_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    topic = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    difficulty = Column(String(20), nullable=False, default="Beginner")  # Beginner, Intermediate, Advanced
    prerequisites = Column(Text, default="[]")  # JSON string
    skills_covered = Column(Text, default="[]")  # JSON string
    duration = Column(Float, default=10.0)  # hours
    tags = Column(Text, default="[]")  # JSON string

    interactions = relationship("Interaction", back_populates="course", cascade="all, delete-orphan")
    recommendation_logs = relationship("RecommendationLog", back_populates="course", cascade="all, delete-orphan")

    def get_prerequisites(self):
        try:
            return json.loads(self.prerequisites) if self.prerequisites else []
        except Exception:
            return []

    def get_skills_covered(self):
        try:
            return json.loads(self.skills_covered) if self.skills_covered else []
        except Exception:
            return []

    def get_tags(self):
        try:
            return json.loads(self.tags) if self.tags else []
        except Exception:
            return []


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    learner_id = Column(String(50), ForeignKey("learners.learner_id"), nullable=False, index=True)
    course_id = Column(String(50), ForeignKey("courses.course_id"), nullable=False, index=True)
    rating = Column(Float, default=0.0)  # 1.0 to 5.0
    quiz_score = Column(Float, default=0.0)  # 0.0 to 100.0
    completed = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner", back_populates="interactions")
    course = relationship("Course", back_populates="interactions")


class SkillProfile(Base):
    __tablename__ = "skill_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(String(50), ForeignKey("learners.learner_id"), nullable=False, index=True)
    topic = Column(String(100), nullable=False)
    skill_level = Column(Float, default=0.5)  # 0.0 to 1.0
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    learner = relationship("Learner", back_populates="skill_profiles")


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    learner_id = Column(String(50), ForeignKey("learners.learner_id"), nullable=False, index=True)
    course_id = Column(String(50), ForeignKey("courses.course_id"), nullable=False, index=True)
    hybrid_score = Column(Float, nullable=False)
    explanation_text = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    learner = relationship("Learner", back_populates="recommendation_logs")
    course = relationship("Course", back_populates="recommendation_logs")
