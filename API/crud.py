from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError
from . import models_db, schemas_db
import logging
from .db_handler import setup_logging

# Configura el logging
setup_logging()
logger = logging.getLogger(__name__)

def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate) -> models_db.Questions_passenger_satisfaction:
    try:
        # Validar el esquema con Pydantic
        passenger_data = passenger.dict()
        db_passenger = models_db.Questions_passenger_satisfaction(**passenger_data)
        db.add(db_passenger)
        db.commit()
        db.refresh(db_passenger)
        logger.info(f"Created passenger satisfaction record with id {db_passenger.id}")
        return db_passenger
    except ValidationError as e:
        logger.error(f"Validation error creating passenger satisfaction record: {str(e)}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"Error creating passenger satisfaction record: {str(e)}")
        db.rollback()
        raise

def get_passenger_satisfaction(db: Session, passenger_id: int) -> models_db.Questions_passenger_satisfaction | None:
    try:
        passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
        if passenger:
            logger.info(f"Retrieved passenger satisfaction record with id {passenger_id}")
        else:
            logger.info(f"No passenger satisfaction record found with id {passenger_id}")
        return passenger
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving passenger satisfaction record: {str(e)}")
        raise

def get_all_passenger_satisfaction(db: Session, skip: int = 0, limit: int = 100) -> list[models_db.Questions_passenger_satisfaction]:
    try:
        passengers = db.query(models_db.Questions_passenger_satisfaction).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(passengers)} passenger satisfaction records with skip {skip} and limit {limit}")
        return passengers
    except SQLAlchemyError as e:
        logger.error(f"Error retrieving all passenger satisfaction records: {str(e)}")
        raise

def update_passenger_satisfaction(db: Session, passenger_id: int, passenger: schemas_db.Questions_passenger_satisfactionUpdate) -> models_db.Questions_passenger_satisfaction | None:
    try:
        db_passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
        
        if not db_passenger:
            logger.info(f"Attempted to update non-existent passenger with id {passenger_id}")
            return None
        
        update_data = passenger.dict(exclude_unset=True)
        for key, value in update_data.items():
            if value is not None:  # Solo actualizar si el valor no es None
                setattr(db_passenger, key, value)
        
        db.commit()
        db.refresh(db_passenger)
        logger.info(f"Updated passenger satisfaction record with id {passenger_id}")
        return db_passenger
    except ValidationError as e:
        logger.error(f"Validation error updating passenger satisfaction record: {str(e)}")
        raise
    except SQLAlchemyError as e:
        logger.error(f"Error updating passenger satisfaction record: {str(e)}")
        db.rollback()
        raise

def delete_passenger_satisfaction(db: Session, passenger_id: int) -> models_db.Questions_passenger_satisfaction | None:
    try:
        db_passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
        
        if not db_passenger:
            logger.info(f"Attempted to delete non-existent passenger with id {passenger_id}")
            return None
        
        db.delete(db_passenger)
        db.commit()
        logger.info(f"Deleted passenger satisfaction record with id {passenger_id}")
        return db_passenger
    except SQLAlchemyError as e:
        logger.error(f"Error deleting passenger satisfaction record: {str(e)}")
        db.rollback()
        raise
