from .database import Base, engine, get_db
from .models_db import Questions_passenger_satisfaction
from .schemas_db import Questions_passenger_satisfactionCreate, Questions_passenger_satisfaction
from .crud import create_passenger_satisfaction, get_passenger_satisfaction, get_all_passenger_satisfaction


Base.metadata.create_all(bind=engine)

from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(
        title="Passenger Satisfaction API",
        description="API para manejar datos de satisfacción de pasajeros",
        version="1.0.0",
    )
    
   
    
    return app