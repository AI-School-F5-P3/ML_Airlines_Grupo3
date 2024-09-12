from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import models_db, schemas_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate, predicted_satisfaction: str | None = None):
    db_passenger = models_db.Questions_passenger_satisfaction(**passenger.model_dump(), predicted_satisfaction=predicted_satisfaction)
    db.add(db_passenger)
    db.commit()
    db.refresh(db_passenger)
    return db_passenger

def get_passenger_satisfaction(db: Session, passenger_id: int):
    try:
        return db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving passenger satisfaction record: {str(e)}")
        raise

def get_all_passenger_satisfaction(db: Session, skip: int = 0, limit: int = 100):
    try:
        return db.query(models_db.Questions_passenger_satisfaction).offset(skip).limit(limit).all()
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving all passenger satisfaction records: {str(e)}")
        raise

def update_passenger_satisfaction(db: Session, passenger_id: int, passenger: schemas_db.Questions_passenger_satisfactionUpdate):
    try:
        db_passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
        if db_passenger:
            update_data = passenger.dict(exclude_unset=True)
            for key, value in update_data.items():
                setattr(db_passenger, key, value)
            db.commit()
            db.refresh(db_passenger)
            logger.info(f"Updated passenger satisfaction record with id {passenger_id}")
        return db_passenger
    except SQLAlchemyError as e:
        logger.error(f"Error updating passenger satisfaction record: {str(e)}")
        db.rollback()
        raise

def delete_passenger_satisfaction(db: Session, passenger_id: int):
    try:
        db_passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
        if db_passenger:
            db.delete(db_passenger)
            db.commit()
            logger.info(f"Deleted passenger satisfaction record with id {passenger_id}")
        return db_passenger
    except SQLAlchemyError as e:
        logger.error(f"Error deleting passenger satisfaction record: {str(e)}")
        db.rollback()
        raise