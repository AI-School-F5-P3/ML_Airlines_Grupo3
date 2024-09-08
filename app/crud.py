from sqlalchemy.orm import Session
from . import models_db, schemas_db

# Función para crear un nuevo registro de satisfacción del pasajero
def create_passenger_satisfaction(db: Session, passenger: schemas_db.Questions_passenger_satisfactionCreate):
    db_passenger = models_db.Questions_passenger_satisfaction(**passenger.dict())
    db.add(db_passenger)
    db.commit()
    db.refresh(db_passenger)
    return db_passenger

# Función para obtener la satisfacción de un pasajero por su ID
def get_passenger_satisfaction(db: Session, passenger_id: int):
    return db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()

# Función para obtener todas las satisfacciones de pasajeros con paginación
def get_all_passenger_satisfaction(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models_db.Questions_passenger_satisfaction).offset(skip).limit(limit).all()

# Función para actualizar la satisfacción de un pasajero por su ID
def update_passenger_satisfaction(db: Session, passenger_id: int, passenger: schemas_db.Questions_passenger_satisfactionUpdate):
    db_passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
    
    if not db_passenger:  # Si no se encuentra el pasajero, devuelve None
        return None
    
    update_data = passenger.dict(exclude_unset=True)  # Solo actualiza los campos enviados
    for key, value in update_data.items():
        setattr(db_passenger, key, value)  # Actualiza los atributos del objeto
    
    db.commit()  # Guarda los cambios en la base de datos
    db.refresh(db_passenger)  # Refresca el objeto con los nuevos datos de la base de datos
    return db_passenger

# Función para eliminar la satisfacción de un pasajero por su ID
def delete_passenger_satisfaction(db: Session, passenger_id: int):
    db_passenger = db.query(models_db.Questions_passenger_satisfaction).filter(models_db.Questions_passenger_satisfaction.id == passenger_id).first()
    
    if not db_passenger:  # Si no se encuentra el pasajero, devuelve None
        return None
    
    db.delete(db_passenger)  # Elimina el registro
    db.commit()  # Confirma los cambios en la base de datos
    return db_passenger
