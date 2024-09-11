from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import models_db, schemas_db
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate) -> models_db.Questions_passenger_satisfaction:
    try:
        db_passenger = models_db.Questions_passenger_satisfaction(**passenger.dict())
        db.add(db_passenger)
        db.commit()
        db.refresh(db_passenger)
        logger.info(f"Created passenger satisfaction record with id {db_passenger.id}")
        return db_passenger
    except SQLAlchemyError as e:
        logger.error(f"Error creating passenger satisfaction record: {str(e)}")
        db.rollback()
        raise


     


