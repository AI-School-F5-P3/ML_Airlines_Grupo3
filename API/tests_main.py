import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from main import app, model
import models_db
import schemas_db
from crud import create_passenger_satisfaction, get_passenger_satisfaction

# Set up test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_create_passenger_satisfaction(db_session):
    passenger_data = {
        "gender": 1,
        "customer_type": "Loyal Customer",
        "age": 30,
        "travel_type": "Business Travel",
        "trip_class": "Eco",
        "flight_distance": 1000,
        "inflight_wifi_service": 3,
        "departure_arrival_time_convenient": 4,
        "online_booking": 5,
        "gate_location": 3,
        "food_and_drink": 4,
        "online_boarding": 3,
        "seat_comfort": 4,
        "inflight_entertainment": 5,
        "onboard_service": 4,
        "leg_room_service": 3,
        "baggage_handling": 4,
        "checkin_service": 5,
        "inflight_service": 4,
        "cleanliness": 5,
        "departure_delay_in_minutes": 10,
        "satisfaction": "Satisfied"
    }
    passenger = schemas_db.Questions_passenger_satisfactionCreate(**passenger_data)
    db_passenger = create_passenger_satisfaction(db_session, passenger)
    assert db_passenger.id is not None
    assert db_passenger.gender == passenger_data["gender"]
    assert db_passenger.satisfaction == passenger_data["satisfaction"]

def test_get_passenger_satisfaction(db_session):
    # First, create a passenger
    passenger_data = {
        "gender": 0,
        "customer_type": "Disloyal Customer",
        "age": 25,
        "travel_type": "Personal Travel",
        "trip_class": "Eco Plus",
        "flight_distance": 800,
        "inflight_wifi_service": 2,
        "departure_arrival_time_convenient": 3,
        "online_booking": 4,
        "gate_location": 2,
        "food_and_drink": 3,
        "online_boarding": 2,
        "seat_comfort": 3,
        "inflight_entertainment": 4,
        "onboard_service": 3,
        "leg_room_service": 2,
        "baggage_handling": 3,
        "checkin_service": 4,
        "inflight_service": 3,
        "cleanliness": 4,
        "departure_delay_in_minutes": 5,
        "satisfaction": "Neutral or Dissatisfied"
    }
    passenger = schemas_db.Questions_passenger_satisfactionCreate(**passenger_data)
    db_passenger = create_passenger_satisfaction(db_session, passenger)
    
    # Now, retrieve the passenger
    retrieved_passenger = get_passenger_satisfaction(db_session, db_passenger.id)
    assert retrieved_passenger is not None
    assert retrieved_passenger.id == db_passenger.id
    assert retrieved_passenger.gender == passenger_data["gender"]
    assert retrieved_passenger.satisfaction == passenger_data["satisfaction"]

def test_submit_and_predict():
    passenger_data = {
        "gender": 1,
        "customer_type": "Loyal Customer",
        "age": 30,
        "travel_type": "Business Travel",
        "trip_class": "Eco",
        "flight_distance": 1000,
        "inflight_wifi_service": 3,
        "departure_arrival_time_convenient": 4,
        "online_booking": 5,
        "gate_location": 3,
        "food_and_drink": 4,
        "online_boarding": 3,
        "seat_comfort": 4,
        "inflight_entertainment": 5,
        "onboard_service": 4,
        "leg_room_service": 3,
        "baggage_handling": 4,
        "checkin_service": 5,
        "inflight_service": 4,
        "cleanliness": 5,
        "departure_delay_in_minutes": 10,
        "satisfaction": "Satisfied"
    }
    response = client.post("/submit_and_predict/", json=passenger_data)
    assert response.status_code == 200
    result = response.json()
    assert "id" in result
    assert "predicted_satisfaction" in result
    assert result["predicted_satisfaction"] in ["Satisfied", "Neutral or Dissatisfied"]

def test_model_loaded():
    assert model is not None, "Model should be loaded"

if __name__ == "__main__":
    pytest.main(["-v", "test_main.py"])
    