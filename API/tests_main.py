import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from database import Base, get_db
from main import app, predict_satisfaction
from models_db import Questions_passenger_satisfaction
from schemas_db import CustomerType, TravelType, TripClass, Satisfaction, Questions_passenger_satisfactionCreate
from crud import create_passenger_satisfaction, get_passenger_satisfaction, update_passenger_satisfaction, delete_passenger_satisfaction
from unittest.mock import patch
from typing import Generator
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la base de datos de prueba en memoria
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Sobrescribir la dependencia de la base de datos
def override_get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="module")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_db) -> Generator[Session, None, None]:
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(autouse=True)
def clear_database(db_session: Session):
    for table in reversed(Base.metadata.sorted_tables):
        db_session.execute(table.delete())
    db_session.commit()

@pytest.fixture(autouse=True)
def mock_model():
    with patch("main.joblib.load") as mock_load:
        mock_model = mock_load.return_value
        mock_model.predict.return_value = [1]  # Simula una predicción "Satisfied"
        yield mock_model

def create_test_passenger(db: Session) -> Questions_passenger_satisfaction:
    passenger = Questions_passenger_satisfactionCreate(
        gender=0,
        customer_type=CustomerType.LOYAL,
        age=30,
        travel_type=TravelType.BUSINESS,
        trip_class=TripClass.BUSINESS,
        flight_distance=1000,
        inflight_wifi_service=4,
        departure_arrival_time_convenient=3,
        online_booking=4,
        gate_location=3,
        food_and_drink=4,
        online_boarding=5,
        seat_comfort=4,
        inflight_entertainment=5,
        onboard_service=4,
        leg_room_service=4,
        baggage_handling=5,
        checkin_service=4,
        inflight_service=5,
        cleanliness=4,
        departure_delay_in_minutes=10,
        satisfaction=Satisfaction.SATISFIED
    )
    return create_passenger_satisfaction(db, passenger)

class TestPassengerSatisfaction:
    def test_create_passenger(self, db_session):
        response = client.post("/submit_and_predict/", json={
            "gender": 0,
            "customer_type": CustomerType.LOYAL.value,
            "age": 30,
            "travel_type": TravelType.BUSINESS.value,
            "trip_class": TripClass.BUSINESS.value,
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
            "baggage_handling": 5,
            "checkin_service": 4,
            "inflight_service": 5,
            "cleanliness": 4,
            "departure_delay_in_minutes": 10,
            "satisfaction": Satisfaction.SATISFIED.value
        })
        assert response.status_code == 200, f"Response: {response.json()}"
        data = response.json()
        assert "id" in data
        assert data["predicted_satisfaction"] == Satisfaction.SATISFIED.value

    def test_get_passenger(self, db_session):
        passenger = create_test_passenger(db_session)
        response = client.get(f"/passengers/{passenger.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == passenger.id
        assert data["age"] == 30

    def test_update_passenger(self, db_session):
        passenger = create_test_passenger(db_session)
        response = client.put(f"/passengers/{passenger.id}", json={"age": 35})
        assert response.status_code == 200
        data = response.json()
        assert data["age"] == 35

    def test_delete_passenger(self, db_session):
        passenger = create_test_passenger(db_session)
        response = client.delete(f"/passengers/{passenger.id}")
        assert response.status_code == 200
        response = client.get(f"/passengers/{passenger.id}")
        assert response.status_code == 404

    def test_get_passengers_pagination(self, db_session: Session):
        # Limpiar la base de datos antes de la prueba
        db_session.query(Questions_passenger_satisfaction).delete()
        db_session.commit()
        logger.info("Database cleaned before pagination test")

        # Crear 15 pasajeros de prueba
        created_passengers = []
        for i in range(15):
            passenger = create_test_passenger(db_session)
            created_passengers.append(passenger)
            logger.info(f"Created test passenger {i+1} with id {passenger.id}")
        
        db_session.commit()

        # Verificar que realmente se crearon 15 pasajeros
        all_passengers = db_session.query(Questions_passenger_satisfaction).all()
        assert len(all_passengers) == 15, f"Expected 15 passengers, but found {len(all_passengers)}"
        logger.info(f"Verified that {len(all_passengers)} passengers were created")

        # Probar la primera página
        response = client.get("/passengers/?skip=0&limit=10")
        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        data = response.json()
        assert len(data) == 10, f"Expected 10 passengers in the first page, but got {len(data)}"
        logger.info(f"First page returned {len(data)} passengers")

        # Probar la segunda página
        response = client.get("/passengers/?skip=10&limit=10")
        assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
        data = response.json()
        assert len(data) == 5, f"Expected 5 passengers in the second page, but got {len(data)}"
        logger.info(f"Second page returned {len(data)} passengers")

        # Verificar el contenido de las páginas
        first_page = client.get("/passengers/?skip=0&limit=10").json()
        second_page = client.get("/passengers/?skip=10&limit=10").json()
        all_retrieved_passengers = first_page + second_page
        assert len(all_retrieved_passengers) == 15, f"Expected to retrieve 15 passengers in total, but got {len(all_retrieved_passengers)}"
        logger.info(f"Retrieved {len(all_retrieved_passengers)} passengers in total from both pages")

        # Verificar que los IDs son únicos y coinciden con los creados
        retrieved_ids = [p['id'] for p in all_retrieved_passengers]
        created_ids = [p.id for p in created_passengers]
        assert set(retrieved_ids) == set(created_ids), "Retrieved passenger IDs do not match created passenger IDs"
        logger.info("Verified that retrieved passenger IDs match created passenger IDs")

        # Verificar el orden de los pasajeros
        assert retrieved_ids == sorted(retrieved_ids), "Passengers are not returned in order of ID"
        logger.info("Verified that passengers are returned in order of ID")

    @patch("main.predict_satisfaction", return_value="Satisfied")
    def test_predict_satisfaction(self, mock_predict, db_session):
        response = client.post("/submit_and_predict/", json={
            "gender": 0,
            "customer_type": CustomerType.LOYAL.value,
            "age": 30,
            "travel_type": TravelType.BUSINESS.value,
            "trip_class": TripClass.BUSINESS.value,
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
            "baggage_handling": 5,
            "checkin_service": 4,
            "inflight_service": 5,
            "cleanliness": 4,
            "departure_delay_in_minutes": 10,
            "satisfaction": Satisfaction.SATISFIED.value
        })
        assert response.status_code == 200, f"Response: {response.json()}"
        data = response.json()
        assert data["predicted_satisfaction"] == Satisfaction.SATISFIED.value

    def test_reload_model(self, mock_model):
        response = client.post("/reload_model/")
        assert response.status_code == 200
        assert response.json() == {"detail": "Model reloaded successfully"}

    def test_invalid_passenger_data(self, db_session):
        response = client.post("/submit_and_predict/", json={
            "gender": 3,  # Invalid gender
            "customer_type": CustomerType.LOYAL.value,
            "age": -5,  # Invalid age
            "travel_type": TravelType.BUSINESS.value,
            "trip_class": TripClass.BUSINESS.value,
            "flight_distance": -100,  # Invalid distance
            "inflight_wifi_service": 6,  # Invalid rating
            "departure_arrival_time_convenient": 3,
            "online_booking": 4,
            "gate_location": 3,
            "food_and_drink": 4,
            "online_boarding": 5,
            "seat_comfort": 4,
            "inflight_entertainment": 5,
            "onboard_service": 4,
            "leg_room_service": 4,
            "baggage_handling": 5,
            "checkin_service": 4,
            "inflight_service": 5,
            "cleanliness": 4,
            "departure_delay_in_minutes": -10,  # Invalid delay
            "satisfaction": "Invalid"  # Invalid satisfaction
        })
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"] == ["body", "gender"] for error in errors)
        assert any(error["loc"] == ["body", "age"] for error in errors)
        assert any(error["loc"] == ["body", "flight_distance"] for error in errors)
        assert any(error["loc"] == ["body", "inflight_wifi_service"] for error in errors)
        assert any(error["loc"] == ["body", "departure_delay_in_minutes"] for error in errors)
        assert any(error["loc"] == ["body", "satisfaction"] for error in errors)

# Puedes agregar más pruebas aquí según sea necesario