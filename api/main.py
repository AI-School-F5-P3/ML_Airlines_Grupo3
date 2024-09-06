from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import crud, models_db, schemas_db
from .database import engine, get_db

# Crear las tablas en la base de datos
models.Base.metadata.create_all(bind=engine)

# Crear la aplicación FastAPI
app = FastAPI()

@app.post("/submit/", response_model=schemas_db.Questions_passenger_satisfaction)
def submit_form(passenger: schemas_db.Questions_passenger_satisfactionCreate, db: Session = Depends(get_db)):
   
    return crud.create_passenger_satisfaction(db=db, passenger=passenger)

@app.get("/passengers/", response_model=list[schemas.Questions_passenger_satisfaction])
def read_passengers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    passengers = crud.get_all_passenger_satisfaction(db, skip=skip, limit=limit)
    return passengers

@app.get("/passengers/{passenger_id}", response_model=schemas.Questions_passenger_satisfaction)
def read_passenger(passenger_id: int, db: Session = Depends(get_db)):
   
    db_passenger = crud.get_passenger_satisfaction(db, passenger_id=passenger_id)
    if db_passenger is None:
        raise HTTPException(status_code=404, detail="Passenger not found")
    return db_passenger

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



