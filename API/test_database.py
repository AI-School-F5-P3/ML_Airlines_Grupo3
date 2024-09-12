import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from main import app
from schemas_db import CustomerType, TravelType, TripClass, Satisfaction

# Setup test database
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

def test_create_and_retrieve_passenger():
    # Test data
    test_passenger = {
        "gender": 1,
        "customer_type": CustomerType.LOYAL,
        "age": 30,
        "travel_type": TravelType.BUSINESS,
        "trip_class": TripClass.BUSINESS,
        "flight_distance": 1000,
        "inflight_wifi_service": 4,
        "departure_arrival_time_convenient": 3,
        "online_booking": 4,
        "gate_location": 3,
        "food_and_drink": 4,
        "online_boarding": 5,
        "seat_comfort": 4,
        "inflight_entertainment": 5,
        "onboard_service": 4,
        "leg_room_service": 4,
        "baggage_handling": 4,
        "checkin_service": 4,
        "inflight_service": 4,
        "cleanliness": 5,
        "departure_delay_in_minutes": 10,
        "satisfaction": Satisfaction.SATISFIED
    }

    # Create passenger
    response = client.post("/submit_and_predict/", json=test_passenger)
    assert response.status_code == 200
    created_passenger = response.json()
    assert "id" in created_passenger

    # Retrieve passenger
    passenger_id = created_passenger["id"]
    response = client.get(f"/passengers/{passenger_id}")
    assert response.status_code == 200
    retrieved_passenger = response.json()

    # Check if retrieved data matches submitted data
    for key, value in test_passenger.items():
        assert retrieved_passenger[key] == value

def test_get_all_passengers():
    response = client.get("/passengers/")
    assert response.status_code == 200
    passengers = response.json()
    assert isinstance(passengers, list)
    assert len(passengers) > 0

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])