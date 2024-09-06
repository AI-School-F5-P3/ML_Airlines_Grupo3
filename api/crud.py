from sqlalchemy.orm import Session
from . import models_db, schemas_db

def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate):
    """
    Crea un nuevo registro de satisfacción de pasajero en la base de datos.
    
    :param db: Sesión de la base de datos
    :param passenger: Datos del pasajero a crear
    :return: El objeto creado
    """
    db_passenger = models_db.Questions_passenger_satisfaction(**passenger.dict())
    db.add(db_passenger)  # Añade el nuevo objeto a la sesión
    db.commit()  # Guarda los cambios en la base de datos
    db.refresh(db_passenger)  # Actualiza el objeto con datos de la base de datos (ej: id generado)
    return db_passenger

def get_passenger_satisfaction(db: Session, passenger_id: int):
   
    return db.query(models.Questions_passenger_satisfaction).filter(models.Questions_passenger_satisfaction.id == passenger_id).first()

def get_all_passenger_satisfaction(db: Session, skip: int = 0, limit: int = 100):
  
    return db.query(models.Questions_passenger_satisfaction).offset(skip).limit(limit).all()



