from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session
from database import Base, engine, SessionLocal

app = FastAPI()

class QuestionsPassengerSatisfaction(Base):
    __tablename__ = "questions_passenger_satisfaction"

    id = Column(Integer, primary_key=True, index=True)
    gender = Column(Integer)
    customer_type = Column(String)
    age = Column(Integer)
    travel_type = Column(String)
    trip_class = Column(String)
    flight_distance = Column(Integer)
    inflight_wifi_service = Column(Integer)
    departure_arrival_time_convenient = Column(Integer)
    online_booking = Column(Integer)
    gate_location = Column(Integer)
    food_and_drink = Column(Integer)
    online_boarding = Column(Integer)
    seat_comfort = Column(Integer)
    inflight_entertainment = Column(Integer)
    onboard_service = Column(Integer)
    leg_room_service = Column(Integer)
    baggage_handling = Column(Integer)
    checkin_service = Column(Integer)
    inflight_service = Column(Integer)
    cleanliness = Column(Integer)
    departure_delay_in_minutes = Column(Integer)
    arrival_delay_in_minutes = Column(Integer)

Base.metadata.create_all(bind=engine)

class QuestionsPassengerSatisfactionCreate(BaseModel):
    gender: int
    customer_type: str
    age: int
    travel_type: str
    trip_class: str
    flight_distance: int
    inflight_wifi_service: int
    departure_arrival_time_convenient: int
    online_booking: int
    gate_location: int
    food_and_drink: int
    online_boarding: int
    seat_comfort: int
    inflight_entertainment: int
    onboard_service: int
    leg_room_service: int
    baggage_handling: int
    checkin_service: int
    inflight_service: int
    cleanliness: int
    departure_delay_in_minutes: int
    arrival_delay_in_minutes: int

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/questions/")
def create_question(question: QuestionsPassengerSatisfactionCreate, db: Session = Depends(get_db)):
    db_question = QuestionsPassengerSatisfaction(**question.dict())
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question