import json
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from db.models import Learner, Interaction, SkillProfile, Course

class ProfileService:
    @staticmethod
    def get_learner(db: Session, learner_id: str) -> Optional[Learner]:
        return db.query(Learner).filter(Learner.learner_id == learner_id).first()

    @staticmethod
    def get_all_learners(db: Session) -> List[Learner]:
        return db.query(Learner).all()

    @staticmethod
    def get_learner_profile_details(db: Session, learner_id: str) -> Optional[Dict[str, Any]]:
        learner = ProfileService.get_learner(db, learner_id)
        if not learner:
            return None

        # Fetch interactions
        interactions = (
            db.query(Interaction)
            .filter(Interaction.learner_id == learner_id)
            .order_by(Interaction.timestamp.desc())
            .all()
        )

        completed_courses = []
        in_progress_courses = []
        quiz_scores = []
        ratings = []

        for it in interactions:
            course = db.query(Course).filter(Course.course_id == it.course_id).first()
            c_info = {
                "course_id": it.course_id,
                "name": course.name if course else it.course_id,
                "topic": course.topic if course else "Unknown",
                "rating": it.rating,
                "quiz_score": it.quiz_score,
                "completed": it.completed,
                "timestamp": it.timestamp.isoformat() if it.timestamp else None
            }
            if it.completed:
                completed_courses.append(c_info)
            else:
                in_progress_courses.append(c_info)
            
            if it.quiz_score > 0:
                quiz_scores.append(it.quiz_score)
            if it.rating > 0:
                ratings.append(it.rating)

        # Skill profile
        skill_profiles = (
            db.query(SkillProfile)
            .filter(SkillProfile.learner_id == learner_id)
            .all()
        )
        skills_map = {sp.topic: sp.skill_level for sp in skill_profiles}

        return {
            "learner_id": learner.learner_id,
            "name": learner.name,
            "education_background": learner.education_background,
            "interests": learner.get_interests(),
            "skills": learner.get_skills(),
            "preferred_topics": learner.get_preferred_topics(),
            "completed_courses": completed_courses,
            "in_progress_courses": in_progress_courses,
            "skill_profile": skills_map,
            "average_quiz_score": round(sum(quiz_scores) / len(quiz_scores), 1) if quiz_scores else 0.0,
            "average_rating": round(sum(ratings) / len(ratings), 1) if ratings else 0.0,
            "total_interactions": len(interactions)
        }

    @staticmethod
    def create_learner(db: Session, learner_id: str, name: str, interests: List[str],
                       education_background: Optional[str], skills: List[str], preferred_topics: List[str]) -> Learner:
        learner = Learner(
            learner_id=learner_id,
            name=name,
            interests=json.dumps(interests),
            education_background=education_background,
            skills=json.dumps(skills),
            preferred_topics=json.dumps(preferred_topics)
        )
        db.add(learner)
        db.commit()
        db.refresh(learner)
        return learner

    @staticmethod
    def update_skill_level(db: Session, learner_id: str, topic: str, new_level: float):
        sp = db.query(SkillProfile).filter(
            SkillProfile.learner_id == learner_id,
            SkillProfile.topic == topic
        ).first()

        if sp:
            sp.skill_level = min(1.0, max(0.0, new_level))
        else:
            sp = SkillProfile(
                learner_id=learner_id,
                topic=topic,
                skill_level=min(1.0, max(0.0, new_level))
            )
            db.add(sp)
        db.commit()
