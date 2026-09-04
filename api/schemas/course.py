from typing import List, Optional
from pydantic import BaseModel

class CourseBase(BaseModel):
    course_id: str
    name: str
    topic: str
    description: str
    difficulty: str = "Beginner"
    prerequisites: List[str] = []
    skills_covered: List[str] = []
    duration: float = 10.0
    tags: List[str] = []

class CourseCreate(CourseBase):
    pass

class CourseResponse(CourseBase):
    class Config:
        from_attributes = True
