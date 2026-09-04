import sys
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from db.database import get_db
from src.catalog.catalog_service import CatalogService
from api.schemas.course import CourseResponse, CourseCreate

router = APIRouter(prefix="/courses", tags=["Courses"])

@router.get("/", response_model=List[CourseResponse])
def list_courses(
    topic: Optional[str] = Query(None, description="Filter by topic"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty (Beginner, Intermediate, Advanced)"),
    search: Optional[str] = Query(None, description="Search term in title, description, or tags"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """List and search courses from the course catalog."""
    courses = CatalogService.list_courses(db, topic=topic, difficulty=difficulty, search=search, limit=limit)
    return [
        CourseResponse(
            course_id=c.course_id,
            name=c.name,
            topic=c.topic,
            description=c.description,
            difficulty=c.difficulty,
            prerequisites=c.get_prerequisites(),
            skills_covered=c.get_skills_covered(),
            duration=c.duration,
            tags=c.get_tags()
        )
        for c in courses
    ]

@router.get("/{course_id}", response_model=CourseResponse)
def get_course(course_id: str, db: Session = Depends(get_db)):
    """Fetch details of a specific course."""
    c = CatalogService.get_course(db, course_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Course '{course_id}' not found.")
    return CourseResponse(
        course_id=c.course_id,
        name=c.name,
        topic=c.topic,
        description=c.description,
        difficulty=c.difficulty,
        prerequisites=c.get_prerequisites(),
        skills_covered=c.get_skills_covered(),
        duration=c.duration,
        tags=c.get_tags()
    )

@router.post("/", response_model=CourseResponse)
def add_or_update_course(payload: CourseCreate, db: Session = Depends(get_db)):
    """Add a new course or update an existing course in the catalog."""
    course = CatalogService.create_or_update_course(
        db=db,
        course_id=payload.course_id,
        name=payload.name,
        topic=payload.topic,
        description=payload.description,
        difficulty=payload.difficulty,
        prerequisites=payload.prerequisites,
        skills_covered=payload.skills_covered,
        duration=payload.duration,
        tags=payload.tags
    )
    return CourseResponse(
        course_id=course.course_id,
        name=course.name,
        topic=course.topic,
        description=course.description,
        difficulty=course.difficulty,
        prerequisites=course.get_prerequisites(),
        skills_covered=course.get_skills_covered(),
        duration=course.duration,
        tags=course.get_tags()
    )
