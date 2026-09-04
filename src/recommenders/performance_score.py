from typing import List, Dict, Any, Optional
import numpy as np
from db.models import Learner, Course, Interaction, SkillProfile

DIFFICULTY_TARGET_SKILL = {
    "Beginner": 0.30,
    "Intermediate": 0.65,
    "Advanced": 0.90
}

class PerformanceScorer:
    @staticmethod
    def calculate_topic_skill(learner_id: str, topic: str, skill_profiles: List[SkillProfile], interactions: List[Interaction], courses_dict: Dict[str, Course]) -> float:
        """
        Determines the learner's skill level in a topic using explicit SkillProfile
        and historical quiz scores.
        """
        # Check skill profile first
        for sp in skill_profiles:
            if sp.learner_id == learner_id and sp.topic.lower() == topic.lower():
                return sp.skill_level

        # Fallback: compute from topic quiz scores
        topic_quizzes = []
        for it in interactions:
            if it.learner_id == learner_id and it.quiz_score > 0:
                c = courses_dict.get(it.course_id)
                if c and c.topic.lower() == topic.lower():
                    topic_quizzes.append(it.quiz_score)

        if topic_quizzes:
            return min(1.0, max(0.0, (sum(topic_quizzes) / len(topic_quizzes)) / 100.0))

        # Default moderate skill level if no data
        return 0.45

    @staticmethod
    def score(learner: Learner, courses: List[Course], interactions: List[Interaction], skill_profiles: List[SkillProfile]) -> Dict[str, float]:
        """
        Scores courses based on skill-vs-difficulty alignment and prerequisite completion.
        Returns a dict {course_id: score} bounded in [0.0, 1.0].
        """
        courses_dict = {c.course_id: c for c in courses}
        completed_course_ids = {it.course_id for it in interactions if it.learner_id == learner.learner_id and it.completed}

        scores = {}
        for course in courses:
            target_skill = DIFFICULTY_TARGET_SKILL.get(course.difficulty, 0.5)
            learner_skill = PerformanceScorer.calculate_topic_skill(
                learner.learner_id, course.topic, skill_profiles, interactions, courses_dict
            )

            # Skill difference penalty (Gaussian curve)
            diff = abs(learner_skill - target_skill)
            alignment = np.exp(-1.5 * (diff ** 2))

            # Prerequisite bonus / penalty
            prereqs = course.get_prerequisites()
            prereq_score = 1.0
            if prereqs:
                satisfied = sum(1 for p in prereqs if p in completed_course_ids)
                prereq_score = satisfied / len(prereqs)

            # Combined performance score
            final_perf = 0.75 * alignment + 0.25 * prereq_score
            scores[course.course_id] = round(float(np.clip(final_perf, 0.0, 1.0)), 4)

        return scores
