from typing import Dict, Any, List, Optional
from db.models import Learner, Course, Interaction, SkillProfile

class RuleBasedExplainer:
    """
    MVP Default Explainer:
    Ranks the 3 hybrid components (Content, Collaborative, Performance) by weighted magnitude
    and builds clear, evidence-backed natural language explanations.
    """
    @staticmethod
    def explain(
        learner: Learner,
        course: Course,
        score_breakdown: Dict[str, float],
        interactions: List[Interaction],
        skill_profiles: List[SkillProfile]
    ) -> Dict[str, Any]:
        cb = score_breakdown.get("content_score", 0.0)
        cf = score_breakdown.get("collaborative_score", 0.0)
        ps = score_breakdown.get("performance_score", 0.0)
        w_cb = score_breakdown.get("weight_content", 0.4)
        w_cf = score_breakdown.get("weight_collaborative", 0.4)
        w_ps = score_breakdown.get("weight_performance", 0.2)

        contributions = [
            ("Content Relevance", cb * w_cb, cb),
            ("Peer Learning Trends", cf * w_cf, cf),
            ("Skill & Difficulty Match", ps * w_ps, ps)
        ]
        contributions.sort(key=lambda x: x[1], reverse=True)

        learner_interests = [i.lower() for i in learner.get_interests()]
        learner_skills = [s.lower() for s in learner.get_skills()]
        course_tags = [t.lower() for t in course.get_tags()]
        course_skills = [s.lower() for s in course.get_skills_covered()]

        matched_skills = list(set(learner_skills).intersection(set(course_skills)))
        matched_interests = [t for t in course_tags if any(i in t or t in i for i in learner_interests)]
        if not matched_interests and course.topic.lower() in [t.lower() for t in learner.get_preferred_topics()]:
            matched_interests.append(course.topic)

        # Completed prerequisites
        completed_course_ids = {it.course_id for it in interactions if it.learner_id == learner.learner_id and it.completed}
        prereqs = course.get_prerequisites()
        prereqs_met = [p for p in prereqs if p in completed_course_ids]

        # Topic skill level
        topic_skill = 0.5
        for sp in skill_profiles:
            if sp.learner_id == learner.learner_id and sp.topic.lower() == course.topic.lower():
                topic_skill = sp.skill_level
                break

        # Generate explanation points
        reasons = []
        primary_factor = contributions[0][0]

        if primary_factor == "Content Relevance":
            if matched_interests:
                reasons.append(f"matches your focus area in {', '.join(matched_interests[:2])}")
            elif matched_skills:
                reasons.append(f"builds upon your current skills in {', '.join(matched_skills[:2])}")
            else:
                reasons.append(f"is strongly aligned with your profile and learning goals in {course.topic}")

        elif primary_factor == "Peer Learning Trends":
            reasons.append(f"is highly rated by learners with comparable backgrounds in {course.topic}")

        elif primary_factor == "Skill & Difficulty Match":
            skill_pct = int(topic_skill * 100)
            reasons.append(f"fits your current {course.difficulty.lower()} level in {course.topic} (proficiency: {skill_pct}%)")

        # Add secondary supporting factor
        secondary_factor = contributions[1][0]
        if secondary_factor == "Skill & Difficulty Match" and len(reasons) < 2:
            if prereqs_met:
                reasons.append(f"you have satisfied prerequisite courses ({', '.join(prereqs_met)})")
            else:
                reasons.append(f"calibrated for your {course.difficulty.lower()} pace")
        elif secondary_factor == "Content Relevance" and len(reasons) < 2:
            if course.topic in learner.get_preferred_topics():
                reasons.append(f"complements your preferred topic '{course.topic}'")
        elif secondary_factor == "Peer Learning Trends" and len(reasons) < 2:
            reasons.append(f"frequently completed successfully by peers")

        explanation_text = f"Recommended because this course {reasons[0]}"
        if len(reasons) > 1:
            explanation_text += f", and {reasons[1]}."
        else:
            explanation_text += "."

        return {
            "explanation_text": explanation_text,
            "primary_factor": primary_factor,
            "secondary_factor": secondary_factor,
            "factor_contributions": [
                {"name": c[0], "weighted_score": round(c[1], 4), "raw_score": round(c[2], 4)}
                for c in contributions
            ],
            "evidence": {
                "matched_interests": matched_interests,
                "matched_skills": matched_skills,
                "prerequisites_met": prereqs_met,
                "topic_skill_level": round(topic_skill, 2),
                "course_difficulty": course.difficulty
            }
        }
