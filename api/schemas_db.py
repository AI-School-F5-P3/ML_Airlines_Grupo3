from pydantic import BaseModel

class Questions_passenger_satisfactionBase(BaseModel):
    """
    Esquema Pydantic base para la satisfacción del pasajero.
    Define todos los campos comunes.
    """
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
    passenger_satisfaction: str

class Questions_passenger_satisfactionCreate(Questions_passenger_satisfactionBase):
    """
    Esquema Pydantic para la creación de un nuevo registro de satisfacción.
    Hereda todos los campos de Questions_passenger_satisfactionBase.
    No incluye el campo 'id' ya que este será generado por la base de datos.
    """
    pass

class Questions_passenger_satisfaction(Questions_passenger_satisfactionBase):
    """
    Esquema Pydantic para la respuesta completa de satisfacción del pasajero.
    Incluye todos los campos de Questions_passenger_satisfactionBase más el 'id'.
    """
    id: int

    class Config:
        orm_mode = True  # Permite que Pydantic lea los datos incluso si no son dict
                         # Esto es necesario para trabajar con objetos SQLAlchemy