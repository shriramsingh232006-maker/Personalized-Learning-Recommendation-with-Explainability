import sys
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from db.database import SessionLocal, init_db
from db.models import Learner, Course, Interaction, SkillProfile
from src.profile.profile_service import ProfileService
from src.catalog.catalog_service import CatalogService
from src.recommenders.hybrid_engine import HybridEngine
from src.feedback.feedback_service import FeedbackService
from src.evaluation.metrics import RecommendationEvaluator

st.set_page_config(
    page_title="Personalized Learning Recommender + XAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .badge-topic {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-diff {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .explanation-box {
        background-color: #E8F5E9;
        border-left: 4px solid #43A047;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.95rem;
        margin: 8px 0 14px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    init_db()
    return HybridEngine()

engine = get_engine()

def get_db_session():
    return SessionLocal()

# ----------------- SIDEBAR -----------------
st.sidebar.title("🎓 Recommender Settings")

db = get_db_session()
learners = ProfileService.get_all_learners(db)

if not learners:
    st.sidebar.warning("Database is empty! Please run synthetic data generator.")
    if st.sidebar.button("Generate Synthetic Data & Seed DB"):
        from data.synthetic.generate_synthetic_data import save_and_seed
        save_and_seed()
        st.sidebar.success("Database generated successfully! Please refresh.")
        st.rerun()
    st.stop()

learner_options = {f"{l.learner_id} - {l.name}": l.learner_id for l in learners}
selected_label = st.sidebar.selectbox("Select Active Learner:", list(learner_options.keys()))
selected_learner_id = learner_options[selected_label]

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Hybrid Fusion Weights")
alpha = st.sidebar.slider("α (Content-Based)", 0.0, 1.0, float(engine.alpha), 0.05)
beta = st.sidebar.slider("β (Collaborative Filtering)", 0.0, 1.0, float(engine.beta), 0.05)
gamma = st.sidebar.slider("γ (Skill / Performance Match)", 0.0, 1.0, float(engine.gamma), 0.05)

# Normalize weights
total_w = alpha + beta + gamma
if total_w > 0:
    engine.alpha = alpha / total_w
    engine.beta = beta / total_w
    engine.gamma = gamma / total_w

top_k = st.sidebar.slider("Top-K Recommendations:", 3, 20, 10)
filter_completed = st.sidebar.checkbox("Exclude Completed Courses", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("System: Hybrid Recommender (CB + CF + PS) with Explainability (XAI)")

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations & XAI",
    "👤 Learner Profile",
    "📚 Course Catalog",
    "📊 System Evaluation"
])

# ----------------- TAB 1: RECOMMENDATIONS -----------------
with tab1:
    st.header(f"Personalized Recommendations for {selected_label}")
    
    with st.spinner("Generating explainable recommendations..."):
        try:
            recs = engine.recommend(
                db=db,
                learner_id=selected_learner_id,
                k=top_k,
                filter_completed=filter_completed
            )
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            recs = []

    if not recs:
        st.info("No recommendations found matching current criteria.")
    else:
        if recs[0].get("is_cold_start"):
            st.info("ℹ️ **Cold-Start Mode Active**: Learner has fewer than 3 interactions. Weights automatically adapted to Content & Skill matching.")

        for rank, item in enumerate(recs, 1):
            with st.container():
                score_pct = int(item['hybrid_score'] * 100)
                col_info, col_score = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"### {rank}. {item['name']}")
                    st.markdown(
                        f"<span class='badge-topic'>{item['topic']}</span> "
                        f"<span class='badge-diff'>{item['difficulty']}</span> "
                        f"⏱️ <b>{item['duration']} hours</b>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"<div class='explanation-box'>💡 <b>Why Recommended:</b> {item['explanation_text']}</div>", unsafe_allow_html=True)
                
                with col_score:
                    st.metric("Hybrid Match", f"{score_pct}%")
                    st.progress(item['hybrid_score'])
                    st.caption(f"Primary driver: **{item['primary_factor']}**")

                # Detailed XAI Accordion
                with st.expander(f"🔍 Explainability & Factor Attribution for '{item['name']}'"):
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("##### Feature Contribution Breakdown")
                        df_factors = pd.DataFrame(item["factor_contributions"])
                        fig_bar = px.bar(
                            df_factors,
                            x="weighted_score",
                            y="name",
                            orientation="h",
                            color="name",
                            text="weighted_score",
                            labels={"weighted_score": "Weighted Contribution", "name": "Component"},
                            title="Component Weights Contribution"
                        )
                        fig_bar.update_layout(showlegend=False, height=220, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_bar, use_container_width=True)

                    with c2:
                        st.markdown("##### SHAP Feature Attribution (Surrogate XAI)")
                        shap_data = item.get("shap_values", {})
                        df_shap = pd.DataFrame([
                            {"Feature": k.replace("_", " "), "Attribution": v}
                            for k, v in shap_data.items()
                        ])
                        fig_shap = px.bar(
                            df_shap,
                            x="Attribution",
                            y="Feature",
                            orientation="h",
                            color="Attribution",
                            color_continuous_scale="Viridis",
                            title="SHAP Local Feature Impact"
                        )
                        fig_shap.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_shap, use_container_width=True)

                    # Quick Feedback Form
                    st.markdown("---")
                    st.markdown("##### ✍️ Submit Feedback on this Course")
                    fb_col1, fb_col2, fb_col3, fb_col4 = st.columns([1.5, 1.5, 1.5, 1.5])
                    with fb_col1:
                        new_rating = st.slider("Rating (1-5):", 1.0, 5.0, 4.5, 0.5, key=f"rate_{item['course_id']}")
                    with fb_col2:
                        new_quiz = st.number_input("Quiz Score (0-100):", 0.0, 100.0, 85.0, 5.0, key=f"quiz_{item['course_id']}")
                    with fb_col3:
                        is_completed = st.checkbox("Mark as Completed", value=True, key=f"comp_{item['course_id']}")
                    with fb_col4:
                        st.write("")
                        st.write("")
                        if st.button("Save Feedback", key=f"btn_save_{item['course_id']}"):
                            FeedbackService.submit_feedback(
                                db=db,
                                learner_id=selected_learner_id,
                                course_id=item["course_id"],
                                rating=new_rating,
                                quiz_score=new_quiz,
                                completed=is_completed
                            )
                            st.success(f"Feedback recorded! Skill level updated for topic '{item['topic']}'.")
                            st.rerun()

                st.markdown("---")

# ----------------- TAB 2: LEARNER PROFILE -----------------
with tab2:
    profile = ProfileService.get_learner_profile_details(db, selected_learner_id)
    if profile:
        st.header(f"👤 {profile['name']} ({profile['learner_id']})")
        st.caption(f"Education: {profile['education_background']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Interactions", profile["total_interactions"])
        m2.metric("Completed Courses", len(profile["completed_courses"]))
        m3.metric("Avg Quiz Score", f"{profile['average_quiz_score']}%")
        m4.metric("Avg Rating Given", f"{profile['average_rating']} / 5")

        st.markdown("---")
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.subheader("🎯 Stated Preferences & Skills")
            st.markdown(f"**Preferred Topics:** {', '.join(profile['preferred_topics'])}")
            st.markdown(f"**Interests:** {', '.join(profile['interests'])}")
            st.markdown(f"**Skills:** {', '.join(profile['skills'])}")

            st.markdown("##### 🏆 Completed Courses")
            if profile["completed_courses"]:
                for cc in profile["completed_courses"]:
                    st.markdown(f"- **{cc['name']}** ({cc['topic']}) | Quiz: `{cc['quiz_score']}%` | Rating: `★ {cc['rating']}`")
            else:
                st.write("No courses completed yet.")

        with p_col2:
            st.subheader("📈 Topic Mastery Skill Profile")
            skill_map = profile["skill_profile"]
            if skill_map:
                df_sp = pd.DataFrame([{"Topic": k, "Skill Level": v * 100} for k, v in skill_map.items()])
                fig_radar = px.line_polar(df_sp, r="Skill Level", theta="Topic", line_close=True)
                fig_radar.update_traces(fill="toself")
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=320)
                st.plotly_chart(fig_radar, use_container_width=True)

# ----------------- TAB 3: COURSE CATALOG -----------------
with tab3:
    st.header("📚 Explore Course Catalog")
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    
    with f_col1:
        search_query = st.text_input("🔍 Search courses by keyword:")
    with f_col2:
        topic_filter = st.selectbox("Filter by Topic:", ["All"] + [
            "Machine Learning", "Data Science", "Web Development", "Cloud Computing", "Cybersecurity", "Mobile Development"
        ])
    with f_col3:
        diff_filter = st.selectbox("Filter by Difficulty:", ["All", "Beginner", "Intermediate", "Advanced"])

    courses = CatalogService.list_courses(
        db,
        topic=None if topic_filter == "All" else topic_filter,
        difficulty=None if diff_filter == "All" else diff_filter,
        search=search_query if search_query else None
    )

    st.caption(f"Showing {len(courses)} courses:")
    for c in courses:
        with st.expander(f"{c.name} ({c.topic}) - [{c.difficulty}]"):
            st.write(c.description)
            st.markdown(f"**Skills Covered:** {', '.join(c.get_skills_covered())}")
            st.markdown(f"**Prerequisites:** {', '.join(c.get_prerequisites()) if c.get_prerequisites() else 'None'}")
            st.markdown(f"**Duration:** {c.duration} hours | **Tags:** {', '.join(c.get_tags())}")

# ----------------- TAB 4: SYSTEM EVALUATION -----------------
with tab4:
    st.header("📊 Recommender System Offline Evaluation (EV)")
    st.markdown("""
    Evaluation of the hybrid recommender system using standard information retrieval metrics on user interaction test splits:
    - **Precision@K**: Proportion of top-K recommendations relevant to the learner.
    - **Recall@K**: Proportion of relevant courses captured in the top-K recommendations.
    - **NDCG@K**: Normalized Discounted Cumulative Gain accounting for position in the ranking.
    - **MAP@K**: Mean Average Precision measuring top-K ranking quality.
    """)

    eval_k = st.slider("Select K cutoff for evaluation:", 3, 15, 5)
    
    if st.button("Run System Evaluation"):
        with st.spinner("Calculating metrics across interaction split..."):
            metrics = RecommendationEvaluator.evaluate_offline(db=db, hybrid_engine=engine, k=eval_k)
            
            ev1, ev2, ev3, ev4 = st.columns(4)
            ev1.metric(f"Precision@{eval_k}", f"{metrics['precision_at_k'] * 100:.1f}%")
            ev2.metric(f"Recall@{eval_k}", f"{metrics['recall_at_k'] * 100:.1f}%")
            ev3.metric(f"NDCG@{eval_k}", f"{metrics['ndcg_at_k'] * 100:.1f}%")
            ev4.metric(f"MAP@{eval_k}", f"{metrics['map_at_k'] * 100:.1f}%")

            df_metrics = pd.DataFrame([
                {"Metric": f"Precision@{eval_k}", "Score": metrics["precision_at_k"]},
                {"Metric": f"Recall@{eval_k}", "Score": metrics["recall_at_k"]},
                {"Metric": f"NDCG@{eval_k}", "Score": metrics["ndcg_at_k"]},
                {"Metric": f"MAP@{eval_k}", "Score": metrics["map_at_k"]}
            ])
            fig_m = px.bar(df_metrics, x="Metric", y="Score", text="Score", color="Metric", title=f"Recommendation Quality at K={eval_k}")
            fig_m.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True)

db.close()
