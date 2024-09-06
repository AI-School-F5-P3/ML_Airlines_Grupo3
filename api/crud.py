from sqlalchemy.orm import Session
from . import models_db, schemas_db

def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate):
   
    db_passenger = models_db.Questions_passenger_satisfaction(**passenger.dict())
    db.add(db_passenger)  
    db.commit()  
    db.refresh(db_passenger)  
    return db_passenger

def get_passenger_satisfaction(db: Session, passenger_id: int):
   
    return db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()

def get_all_passenger_satisfaction(db: Session, skip: int = 0, limit: int = 100):
  
    return db.query(models_db.Questions_passenger_satisfaction).offset(skip).limit(limit).all()



