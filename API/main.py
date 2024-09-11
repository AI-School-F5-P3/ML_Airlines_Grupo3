from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import models_db, schemas_db
import joblib  
import os
from dotenv import load_dotenv
import crud
from database import engine, Base, get_db

Base.metadata.create_all(bind=engine)



# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Crear las tablas en la base de datos
models_db.Base.metadata.create_all(bind=engine)

# Inicializar la app
app = FastAPI()

# Cargar el modelo de Machine Learning
MODEL_PATH = os.getenv("MODEL_PATH", "default_model_path/model.joblib")
print(f"Trying to load model from: {MODEL_PATH}")


@app.post("/submit/", response_model=schemas_db.Questions_passenger_satisfaction)
def submit_form(passenger: schemas_db.Questions_passenger_satisfactionCreate, db: Session = Depends(get_db)):
    return crud.create_passenger_satisfaction(db=db, passenger=passenger)

@app.get("/passengers/", response_model=list[schemas_db.Questions_passenger_satisfaction])
def read_passengers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    passengers = crud.get_all_passenger_satisfaction(db, skip=skip, limit=limit)
    return passengers

@app.get("/passengers/{passenger_id}", response_model=schemas_db.Questions_passenger_satisfaction)
def read_passenger(passenger_id: int, db: Session = Depends(get_db)):
    db_passenger = crud.get_passenger_satisfaction(db, passenger_id=passenger_id)
    if db_passenger is None:
        raise HTTPException(status_code=404, detail="Passenger not found")
    return db_passenger

@app.post("/predict/")
def predict_satisfaction(passenger: schemas_db.Questions_passenger_satisfactionCreate):
    """
    Endpoint para predecir la satisfacción del pasajero.
    """
    # Preparar los datos para el modelo
    passenger_data = [
        [
            passenger.gender,
            passenger.customer_type,
            passenger.age,
            passenger.travel_type,
            passenger.trip_class,
            passenger.flight_distance,
            passenger.inflight_wifi_service,
            passenger.departure_arrival_time_convenient,
            passenger.online_booking,
            passenger.gate_location,
            passenger.food_and_drink,
            passenger.online_boarding,
            passenger.seat_comfort,
            passenger.inflight_entertainment,
            passenger.onboard_service,
            passenger.leg_room_service,
            passenger.baggage_handling,
            passenger.checkin_service,
            passenger.inflight_service,
            passenger.cleanliness,
            passenger.departure_delay_in_minutes,
            passenger.arrival_delay_in_minutes
        ]
    ]

    # Cargar el modelo de Machine Learning
    model = joblib.load(MODEL_PATH)

    # Hacer la predicción
    try:
        prediction = model.predict(passenger_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
