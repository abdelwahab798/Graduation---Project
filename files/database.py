from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./mediscan.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # مطلوب لـ SQLite مع FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency - بتتحط في كل route محتاج database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
