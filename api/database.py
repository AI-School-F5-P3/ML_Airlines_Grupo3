from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener la URL de la base de datos desde las variables de entorno
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Crear el motor de SQLAlchemy
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Crear una clase de sesión local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Crear una clase base para los modelos declarativos
Base = declarative_base()

# Función para obtener una sesión de base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db  # Usar yield para que FastAPI pueda manejar la sesión como un contexto
    finally:
        db.close()  # Asegurar que la sesión se cierre después de cada solicitud