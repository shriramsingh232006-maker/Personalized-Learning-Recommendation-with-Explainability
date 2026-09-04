from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from db.models import Interaction, Course, SkillProfile, Learner

class FeedbackService:
    @staticmethod
    def submit_feedback(
        db: Session,
        learner_id: str,
        course_id: str,
        rating: Optional[float] = None,
        quiz_score: Optional[float] = None,
        completed: Optional[bool] = None
    ) -> Dict[str, Any]:
        # Validate learner and course
        learner = db.query(Learner).filter(Learner.learner_id == learner_id).first()
        if not learner:
            raise ValueError(f"Learner '{learner_id}' not found.")

        course = db.query(Course).filter(Course.course_id == course_id).first()
        if not course:
            raise ValueError(f"Course '{course_id}' not found.")

        # Check existing interaction
        interaction = db.query(Interaction).filter(
            Interaction.learner_id == learner_id,
            Interaction.course_id == course_id
        ).first()

        if interaction:
            if rating is not None:
                interaction.rating = rating
            if quiz_score is not None:
                interaction.quiz_score = quiz_score
            if completed is not None:
                interaction.completed = completed
            interaction.timestamp = datetime.utcnow()
        else:
            interaction = Interaction(
                learner_id=learner_id,
                course_id=course_id,
                rating=rating if rating is not None else 0.0,
                quiz_score=quiz_score if quiz_score is not None else 0.0,
                completed=completed if completed is not None else False,
                timestamp=datetime.utcnow()
            )
            db.add(interaction)

        db.commit()
        db.refresh(interaction)

        # Trigger skill profile refresh
        updated_skill_level = FeedbackService._refresh_topic_skill(db, learner_id, course.topic)

        return {
            "status": "success",
            "interaction_id": interaction.interaction_id,
            "learner_id": learner_id,
            "course_id": course_id,
            "rating": interaction.rating,
            "quiz_score": interaction.quiz_score,
            "completed": interaction.completed,
            "topic": course.topic,
            "updated_topic_skill_level": updated_skill_level
        }

    @staticmethod
    def _refresh_topic_skill(db: Session, learner_id: str, topic: str) -> float:
        """
        Recalculates learner skill level for the specified topic from all completed
        or quiz-scored interactions in that topic.
        """
        topic_courses = db.query(Course).filter(Course.topic == topic).all()
        topic_cids = {c.course_id for c in topic_courses}

        interactions = db.query(Interaction).filter(
            Interaction.learner_id == learner_id,
            Interaction.course_id.in_(topic_cids)
        ).all()

        if not interactions:
            return 0.5

        scores = [it.quiz_score for it in interactions if it.quiz_score > 0]
        completed_count = sum(1 for it in interactions if it.completed)

        base_level = (sum(scores) / len(scores)) / 100.0 if scores else 0.5
        completion_bonus = min(0.2, completed_count * 0.05)
        new_skill = min(1.0, max(0.1, base_level + completion_bonus))
        new_skill = round(new_skill, 2)

        sp = db.query(SkillProfile).filter(
            SkillProfile.learner_id == learner_id,
            SkillProfile.topic == topic
        ).first()

        if sp:
            sp.skill_level = new_skill
            sp.last_updated = datetime.utcnow()
        else:
            sp = SkillProfile(
                learner_id=learner_id,
                topic=topic,
                skill_level=new_skill,
                last_updated=datetime.utcnow()
            )
            db.add(sp)

        db.commit()
        return new_skill
