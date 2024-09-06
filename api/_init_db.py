from sqlalchemy.orm import Session
from database import SessionLocal, engine
from main import Base, QuestionsPassengerSatisfaction

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()