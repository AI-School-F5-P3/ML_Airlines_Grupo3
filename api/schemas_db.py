from pydantic import BaseModel

class Questions_passenger_satisfactionBase(BaseModel):
  
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
   
    pass

class Questions_passenger_satisfaction(Questions_passenger_satisfactionBase):
   
    id: int

    class Config:
        orm_mode = True  # Permite que Pydantic lea los datos incluso si no son dict
                         # Esto es necesario para trabajar con objetos SQLAlchemy