import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import Base, get_db
from main import app
from schemas_db import CustomerType, TravelType, TripClass, Satisfaction
from models_db import Questions_passenger_satisfaction

# Configuración de la base de datos de prueba en memoria
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Sobrescribir la dependencia de la base de datos en la aplicación
def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

# Ejemplo de un pasajero para usar en las pruebas
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
    "arrival_delay_in_minutes": 0,
    "satisfaction": Satisfaction.SATISFIED
}

@pytest.fixture(scope="session", autouse=True)
def setup_module():
    """ Fixture que crea las tablas una vez por sesión completa de prueba. """
    Base.metadata.create_all(bind=engine)
    print("Tables created:", Base.metadata.tables.keys())  # Depuración
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    """ Fixture que abre una nueva sesión de base de datos por prueba. """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    # Asegurarse de que las tablas existen
    Base.metadata.create_all(bind=connection)
    
    yield session
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(autouse=True)
def clear_tables(db_session):
    """ Limpia las tablas antes de cada prueba para asegurar un estado limpio. """
    for table in reversed(Base.metadata.sorted_tables):
        db_session.execute(table.delete())
    db_session.commit()
    
def test_table_exists(db_session):
    """Verifica que la tabla passenger_satisfaction existe."""
    from sqlalchemy import inspect
    inspector = inspect(engine)
    assert "passenger_satisfaction" in inspector.get_table_names()

# Pruebas agrupadas en una clase
class TestPassengerSatisfaction:

    def test_create_passenger_satisfaction_returns_200_with_valid_data(self, db_session):
        """Prueba la creación de un nuevo pasajero con datos válidos."""
        response = client.post("/submit/", json=example_passenger)
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.SATISFIED
        assert "id" in data

    def test_get_passenger_returns_200_with_valid_id(self, db_session):
        """Prueba la obtención de un pasajero existente por su ID."""
        # Crear un pasajero
        response = client.post("/submit/", json=example_passenger)
        assert response.status_code == 200
        passenger_id = response.json()["id"]
        # Obtener el pasajero por su ID
        response = client.get(f"/passengers/{passenger_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == passenger_id
        assert data["satisfaction"] == Satisfaction.SATISFIED

    def test_update_passenger_satisfaction_returns_200(self, db_session):
        """Prueba la actualización de un pasajero existente."""
        # Crear un pasajero
        response = client.post("/submit/", json=example_passenger)
        assert response.status_code == 200
        passenger_id = response.json()["id"]
        # Actualizar algunos campos
        updated_data = {
            "satisfaction": Satisfaction.NEUTRAL,
            "age": 30
        }
        response = client.put(f"/passengers/{passenger_id}", json=updated_data)
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.NEUTRAL
        assert data["age"] == 30
        assert data["flight_distance"] == example_passenger["flight_distance"]  # Asegurarse de que otros datos no cambian

    def test_non_existent_passenger_returns_404(self, db_session):
        """Prueba la obtención de un pasajero que no existe."""
        response = client.get("/passengers/9999")
        assert response.status_code == 404

    def test_delete_passenger_satisfaction_returns_200(self, db_session):
        """Prueba la eliminación de un pasajero existente."""
        # Crear un pasajero
        response = client.post("/submit/", json=example_passenger)
        assert response.status_code == 200
        passenger_id = response.json()["id"]
        # Eliminar el pasajero
        response = client.delete(f"/passengers/{passenger_id}")
        assert response.status_code == 200
        # Verificar que el pasajero ha sido eliminado
        response = client.get(f"/passengers/{passenger_id}")
        assert response.status_code == 404

    def test_get_all_passengers_returns_200_with_list(self, db_session):
        """Prueba la obtención de todos los pasajeros."""
        # Crear varios pasajeros
        for _ in range(3):
            client.post("/submit/", json=example_passenger)
        response = client.get("/passengers/")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_predict_satisfaction_returns_200_with_prediction(self, db_session):
        """Prueba el endpoint de predicción de satisfacción."""
        response = client.post("/predict/", json=example_passenger)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], str)

    def test_create_invalid_passenger_returns_422(self, db_session):
        """Prueba la creación de un pasajero con datos inválidos."""
        invalid_passenger = example_passenger.copy()
        invalid_passenger["age"] = -5  # Edad inválida
        response = client.post("/submit/", json=invalid_passenger)
        assert response.status_code == 422

    def test_update_non_existent_passenger_returns_404(self, db_session):
        """Prueba la actualización de un pasajero que no existe."""
        response = client.put("/passengers/9999", json={"age": 40})
        assert response.status_code == 404

    def test_delete_non_existent_passenger_returns_404(self, db_session):
        """Prueba la eliminación de un pasajero que no existe."""
        response = client.delete("/passengers/9999")
        assert response.status_code == 404

    def test_get_passengers_with_pagination(self, db_session):
        """Prueba la paginación al obtener pasajeros."""
        # Crear varios pasajeros
        for _ in range(15):
            client.post("/submit/", json=example_passenger)
        
        # Probar la paginación
        response = client.get("/passengers/?skip=5&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_create_passenger_with_minimum_values(self, db_session):
        """Prueba la creación de un pasajero con valores mínimos permitidos."""
        min_passenger = {
            "gender": 0,
            "customer_type": CustomerType.DISLOYAL,
            "age": 18,
            "travel_type": TravelType.PERSONAL,
            "trip_class": TripClass.ECO,
            "flight_distance": 100,
            "inflight_wifi_service": 0,
            "departure_arrival_time_convenient": 0,
            "online_booking": 0,
            "gate_location": 0,
            "food_and_drink": 0,
            "online_boarding": 0,
            "seat_comfort": 0,
            "inflight_entertainment": 0,
            "onboard_service": 0,
            "leg_room_service": 0,
            "baggage_handling": 0,
            "checkin_service": 0,
            "inflight_service": 0,
            "cleanliness": 0,
            "departure_delay_in_minutes": 0,
            "arrival_delay_in_minutes": 0,
            "satisfaction": Satisfaction.NEUTRAL
        }
        response = client.post("/submit/", json=min_passenger)
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.NEUTRAL

    def test_create_passenger_with_maximum_values(self, db_session):
        """Prueba la creación de un pasajero con valores máximos permitidos."""
        max_passenger = {
            "gender": 1,
            "customer_type": CustomerType.LOYAL,
            "age": 120,
            "travel_type": TravelType.BUSINESS,
            "trip_class": TripClass.BUSINESS,
            "flight_distance": 10000,
            "inflight_wifi_service": 5,
            "departure_arrival_time_convenient": 5,
            "online_booking": 5,
            "gate_location": 5,
            "food_and_drink": 5,
            "online_boarding": 5,
            "seat_comfort": 5,
            "inflight_entertainment": 5,
            "onboard_service": 5,
            "leg_room_service": 5,
            "baggage_handling": 5,
            "checkin_service": 5,
            "inflight_service": 5,
            "cleanliness": 5,
            "departure_delay_in_minutes": 1440,  # 24 horas
            "arrival_delay_in_minutes": 1440,  # 24 horas
            "satisfaction": Satisfaction.SATISFIED
        }
        response = client.post("/submit/", json=max_passenger)
        assert response.status_code == 200
        data = response.json()
        assert data["satisfaction"] == Satisfaction.SATISFIED

    def test_partial_update_passenger(self, db_session):
        """Prueba la actualización parcial de un pasajero."""
        # Crear un pasajero
        response = client.post("/submit/", json=example_passenger)
        assert response.status_code == 200
        passenger_id = response.json()["id"]
        
        # Actualizar solo un campo
        response = client.put(f"/passengers/{passenger_id}", json={"age": 35})
        assert response.status_code == 200
        data = response.json()
        assert data["age"] == 35
        assert data["satisfaction"] == Satisfaction.SATISFIED  # Asegurarse de que otros campos no cambian