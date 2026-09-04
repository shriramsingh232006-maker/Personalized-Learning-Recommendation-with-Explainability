import os
try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Find root dir config
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

def get_db_url():
    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")
    if yaml and CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            if cfg and "database" in cfg and "url" in cfg["database"]:
                url = cfg["database"]["url"]
                if url.startswith("sqlite:///./"):
                    db_path = BASE_DIR / url.replace("sqlite:///./", "")
                    return f"sqlite:///{db_path}"
                return url
    return f"sqlite:///{BASE_DIR / 'learning_recommender.db'}"

DATABASE_URL = get_db_url()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from db.models import Learner, Course, Interaction, SkillProfile, RecommendationLog
    Base.metadata.create_all(bind=engine)
