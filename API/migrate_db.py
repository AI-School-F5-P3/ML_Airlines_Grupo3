import sys
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeMeta

# Añadir el directorio actual al PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from database import engine, Base
from models_db import Questions_passenger_satisfaction

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

def create_table_if_not_exists(engine: Engine, base: DeclarativeMeta) -> None:
    """Verifica si la tabla 'passenger_satisfaction' existe y la crea si es necesario."""
    inspector = inspect(engine)
    if not inspector.has_table("passenger_satisfaction"):
        base.metadata.create_all(engine)
        logger.info("Table 'passenger_satisfaction' created.")
    else:
        logger.info("Table 'passenger_satisfaction' already exists.")

def add_column_if_not_exists(engine: Engine) -> None:
    """Verifica si la columna 'predicted_satisfaction' existe y la agrega si es necesario."""
    inspector = inspect(engine)
    columns = inspector.get_columns("passenger_satisfaction")
    column_names = [column['name'] for column in columns]
    
    if "predicted_satisfaction" not in column_names:
        try:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE passenger_satisfaction ADD COLUMN predicted_satisfaction VARCHAR"))
            logger.info("Column 'predicted_satisfaction' added to the table.")
        except Exception as e:
            logger.error(f"Error adding column 'predicted_satisfaction': {str(e)}")
            raise
    else:
        logger.info("Column 'predicted_satisfaction' already exists.")

def migrate() -> None:
    """Ejecuta las migraciones necesarias."""
    create_table_if_not_exists(engine, Base)
    add_column_if_not_exists(engine)

if __name__ == "__main__":
    try:
        migrate()
        logger.info("Migration completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)