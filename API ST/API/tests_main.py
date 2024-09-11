import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from main import app, predict_satisfaction
from schemas_db import CustomerType, TravelType, TripClass, Satisfaction, Questions_passenger_satisfactionCreate
from models_db import Questions_passenger_satisfaction
from unittest.mock import patch

# Configuración de la base de datos de prueba en memoria
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

example_passenger = {
    "gender": 0,
    "customer_type": CustomerType.LOYAL,
    "age": 25,
    "travel_type": TravelType.BUSINESS,
    "trip_class": TripClass.BUSINESS,
    "flight_distance": 1000,
    "inflight_wifi_service": 4,
    "departure_arrival_time_convenient": 3,
    "online_booking": 4,
    "gate_location": 2,
    "food_and_drink": 5,
    "online_boarding": 4,
    "seat_comfort": 5,
    "inflight_entertainment": 3,
    "onboard_service": 5,
    "leg_room_service": 4,
    "baggage_handling": 3,
    "checkin_service": 5,
    "inflight_service": 4,
    "cleanliness": 4,
    "departure_delay_in_minutes": 0,
    "satisfaction": Satisfaction.SATISFIED
}

# Función auxiliar para crear un pasajero
def create_passenger():
    response = client.post("/submit_and_predict/", json=example_passenger)
    assert response.status_code == 200
    return response.json()["id"]

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_db):
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(autouse=True)
def clear_tables(db_session):
    for table in reversed(Base.metadata.sorted_tables):
        db_session.execute(table.delete())
    db_session.commit()

def test_table_exists(db_session):
    inspector = inspect(engine)
    assert "passenger_satisfaction" in inspector.get_table_names()

class TestPassengerSatisfaction:
    def test_create_passenger_satisfaction(self, db_session):
        response = client.post("/submit_and_predict/", json=example_passenger)
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.SATISFIED
        assert "id" in data
        assert "predicted_satisfaction" in data

    def test_create_passenger_satisfaction_invalid_data(self, db_session):
        invalid_passenger = example_passenger.copy()
        invalid_passenger["age"] = "invalid"
        response = client.post("/submit_and_predict/", json=invalid_passenger)
        assert response.status_code == 422

    @patch("main.predict_satisfaction", return_value="Satisfied")
    def test_predict_satisfaction(self, mock_predict, db_session):
        passenger_id = create_passenger()
        response = client.get(f"/passengers/{passenger_id}")
        assert response.status_code == 200
        data = response.json()
        assert "predicted_satisfaction" in data
        assert data["predicted_satisfaction"] == "Satisfied"
        mock_predict.assert_called_once()

    def test_get_passengers(self, db_session):
        create_passenger()
        response = client.get("/passengers/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["satisfaction"] == Satisfaction.SATISFIED

    def test_get_passenger(self, db_session):
        passenger_id = create_passenger()
        response = client.get(f"/passengers/{passenger_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.SATISFIED

    def test_update_passenger_satisfaction(self, db_session):
        passenger_id = create_passenger()
        updated_data = {"satisfaction": Satisfaction.NEUTRAL, "age": 30}
        response = client.put(f"/passengers/{passenger_id}", json=updated_data)
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.NEUTRAL
        assert data["age"] == 30

    def test_delete_passenger_satisfaction(self, db_session):
        passenger_id = create_passenger()
        response = client.delete(f"/passengers/{passenger_id}")
        assert response.status_code == 200
        # Verify the passenger was deleted
        response = client.get(f"/passengers/{passenger_id}")
        assert response.status_code == 404

    def test_get_non_existent_passenger(self, db_session):
        response = client.get("/passengers/99999")
        assert response.status_code == 404

    def test_update_non_existent_passenger(self, db_session):
        updated_data = {"satisfaction": Satisfaction.NEUTRAL, "age": 30}
        response = client.put("/passengers/99999", json=updated_data)
        assert response.status_code == 404

    def test_delete_non_existent_passenger(self, db_session):
        response = client.delete("/passengers/99999")
        assert response.status_code == 404

    @pytest.mark.parametrize("invalid_field,invalid_value", [
        ("age", -1),
        ("gender", 3),
        ("customer_type", "INVALID_TYPE"),
        ("flight_distance", -100),
        ("inflight_wifi_service", 6),
    ])
    def test_create_passenger_satisfaction_invalid_data_parametrized(self, db_session, invalid_field, invalid_value):
        invalid_passenger = example_passenger.copy()
        invalid_passenger[invalid_field] = invalid_value
        response = client.post("/submit_and_predict/", json=invalid_passenger)
        assert response.status_code == 422

    def test_get_passengers_pagination(self, db_session):
        # Create multiple passengers
        for _ in range(15):
            create_passenger()

        # Test first page
        response = client.get("/passengers/?skip=0&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10

        # Test second page
        response = client.get("/passengers/?skip=10&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    @patch("main.joblib.load")
    def test_reload_model(self, mock_joblib_load):
        mock_joblib_load.return_value = "mocked_model"
        response = client.post("/reload_model/")
        assert response.status_code == 200
        assert response.json() == {"detail": "Model reloaded successfully"}
        mock_joblib_load.assert_called_once()

    def test_predict_satisfaction_function():
        passenger_data = [[0, 1, 25, 1, 2, 1000, 4, 3, 4, 2, 5, 4, 5, 3, 5, 4, 3, 5, 4, 4, 0]]
        result = predict_satisfaction(passenger_data)
        assert result in ["Satisfied", "Neutral or Dissatisfied", "Unknown"]

# You can add more tests as needed