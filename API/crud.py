from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import models_db, schemas_db
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate) -> models_db.Questions_passenger_satisfaction:
    try:
        # Convert the Pydantic model to a dictionary
        passenger_dict = passenger.model_dump()
        
        # Extract the predicted_satisfaction if it exists
        predicted_satisfaction = passenger_dict.pop('predicted_satisfaction', None)
        
        logger.info(f"Creating passenger satisfaction record with data: {passenger_dict}")
        logger.info(f"Predicted satisfaction: {predicted_satisfaction}")
        
        # Create the database model instance
        db_passenger = models_db.Questions_passenger_satisfaction(**passenger_dict)
        
        # Set the predicted_satisfaction
        if predicted_satisfaction is not None:
            db_passenger.predicted_satisfaction = predicted_satisfaction
        
        db.add(db_passenger)
        db.commit()
        db.refresh(db_passenger)
        logger.info(f"Created passenger satisfaction record with id {db_passenger.id}")
        logger.info(f"Saved record: {db_passenger.__dict__}")
        return db_passenger
    except SQLAlchemyError as e:
        logger.error(f"Error creating passenger satisfaction record: {str(e)}")
        db.rollback()
        raise


     


