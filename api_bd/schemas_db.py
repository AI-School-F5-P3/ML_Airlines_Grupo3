from pydantic import BaseModel, Field
from enum import Enum

class CustomerType(str, Enum):
    loyal = "Loyal Customer"
    disloyal = "Disloyal Customer"

class TravelType(str, Enum):
    personal = "Personal Travel"
    business = "Business Travel"

class TripClass(str, Enum):
    eco = "Eco"
    eco_plus = "Eco Plus"
    business = "Business"

class PassengerSatisfaction(str, Enum):
    neutral = "Neutral or Dissatisfied"
    satisfied = "Satisfied"

class Questions_passenger_satisfactionBase(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="0 for Female, 1 for Male")
    customer_type: CustomerType
    age: int = Field(..., ge=0, le=120)
    travel_type: TravelType
    trip_class: TripClass
    flight_distance: int = Field(..., ge=0)
    inflight_wifi_service: int = Field(..., ge=0, le=5)
    departure_arrival_time_convenient: int = Field(..., ge=0, le=5)
    online_booking: int = Field(..., ge=0, le=5)
    gate_location: int = Field(..., ge=0, le=5)
    food_and_drink: int = Field(..., ge=0, le=5)
    online_boarding: int = Field(..., ge=0, le=5)
    seat_comfort: int = Field(..., ge=0, le=5)
    inflight_entertainment: int = Field(..., ge=0, le=5)
    onboard_service: int = Field(..., ge=0, le=5)
    leg_room_service: int = Field(..., ge=0, le=5)
    baggage_handling: int = Field(..., ge=0, le=5)
    checkin_service: int = Field(..., ge=0, le=5)
    inflight_service: int = Field(..., ge=0, le=5)
    cleanliness: int = Field(..., ge=0, le=5)
    departure_delay_in_minutes: int = Field(..., ge=0)
    arrival_delay_in_minutes: int = Field(..., ge=0)
    passenger_satisfaction: PassengerSatisfaction

class Questions_passenger_satisfactionCreate(Questions_passenger_satisfactionBase):
    pass

class Questions_passenger_satisfactionUpdate(BaseModel):
    gender: int | None = Field(None, ge=0, le=1, description="0 for Female, 1 for Male")
    customer_type: CustomerType | None = None
    age: int | None = Field(None, ge=0, le=120)
    travel_type: TravelType | None = None
    trip_class: TripClass | None = None
    flight_distance: int | None = Field(None, ge=0)
    inflight_wifi_service: int | None = Field(None, ge=0, le=5)
    departure_arrival_time_convenient: int | None = Field(None, ge=0, le=5)
    online_booking: int | None = Field(None, ge=0, le=5)
    gate_location: int | None = Field(None, ge=0, le=5)
    food_and_drink: int | None = Field(None, ge=0, le=5)
    online_boarding: int | None = Field(None, ge=0, le=5)
    seat_comfort: int | None = Field(None, ge=0, le=5)
    inflight_entertainment: int | None = Field(None, ge=0, le=5)
    onboard_service: int | None = Field(None, ge=0, le=5)
    leg_room_service: int | None = Field(None, ge=0, le=5)
    baggage_handling: int | None = Field(None, ge=0, le=5)
    checkin_service: int | None = Field(None, ge=0, le=5)
    inflight_service: int | None = Field(None, ge=0, le=5)
    cleanliness: int | None = Field(None, ge=0, le=5)
    departure_delay_in_minutes: int | None = Field(None, ge=0)
    arrival_delay_in_minutes: int | None = Field(None, ge=0)
    passenger_satisfaction: PassengerSatisfaction | None = None

class Questions_passenger_satisfaction(Questions_passenger_satisfactionBase):
    id: int

    class Config:
        orm_mode = True