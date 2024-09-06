from sqlalchemy.orm import Session
from main import QuestionsPassengerSatisfaction

def get_questions(db: Session, question_id: int):
    return db.query(QuestionsPassengerSatisfaction).filter(QuestionsPassengerSatisfaction.id == question_id).first()

def create_question(db: Session, question: QuestionsPassengerSatisfaction):
    db.add(question)
    db.commit()
    db.refresh(question)
    return question