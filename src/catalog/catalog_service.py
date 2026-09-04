import json
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from db.models import Course

class CatalogService:
    @staticmethod
    def get_course(db: Session, course_id: str) -> Optional[Course]:
        return db.query(Course).filter(Course.course_id == course_id).first()

    @staticmethod
    def list_courses(
        db: Session,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100
    ) -> List[Course]:
        query = db.query(Course)
        if topic:
            query = query.filter(Course.topic.ilike(f"%{topic}%"))
        if difficulty:
            query = query.filter(Course.difficulty.ilike(f"%{difficulty}%"))
        if search:
            search_fmt = f"%{search}%"
            query = query.filter(
                (Course.name.ilike(search_fmt)) |
                (Course.description.ilike(search_fmt)) |
                (Course.tags.ilike(search_fmt))
            )
        return query.limit(limit).all()

    @staticmethod
    def create_or_update_course(
        db: Session,
        course_id: str,
        name: str,
        topic: str,
        description: str,
        difficulty: str,
        prerequisites: List[str],
        skills_covered: List[str],
        duration: float,
        tags: List[str]
    ) -> Course:
        course = db.query(Course).filter(Course.course_id == course_id).first()
        if course:
            course.name = name
            course.topic = topic
            course.description = description
            course.difficulty = difficulty
            course.prerequisites = json.dumps(prerequisites)
            course.skills_covered = json.dumps(skills_covered)
            course.duration = duration
            course.tags = json.dumps(tags)
        else:
            course = Course(
                course_id=course_id,
                name=name,
                topic=topic,
                description=description,
                difficulty=difficulty,
                prerequisites=json.dumps(prerequisites),
                skills_covered=json.dumps(skills_covered),
                duration=duration,
                tags=json.dumps(tags)
            )
            db.add(course)
        db.commit()
        db.refresh(course)
        return course

    @staticmethod
    def get_course_representation_text(course: Course) -> str:
        """
        Creates a rich semantic text string for TF-IDF / vector embeddings.
        """
        skills = " ".join(course.get_skills_covered())
        tags = " ".join(course.get_tags())
        prereqs = " ".join(course.get_prerequisites())
        return f"{course.name} {course.topic} {course.difficulty} {course.description} {skills} {tags} {prereqs}"
