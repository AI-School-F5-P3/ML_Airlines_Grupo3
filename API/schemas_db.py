from pydantic import BaseModel, Field

class Questions_passenger_satisfaction_Base(BaseModel):
    gender: int = Field(..., ge=0, le=1, description="0 for Female, 1 for Male")  # Integer
    customer_type: int = Field(..., ge=0, le=1, description="0 for Disloyal Customer, 1 for Loyal Customer")  # Integer
    age: int = Field(..., ge=0, le=120)
    travel_type: int = Field(..., ge=0, le=1, description="0 for Personal, 1 for Business")  # Integer
    trip_class: int = Field(..., ge=0, le=2, description="0 for Business, 1 for Eco, 2 for Eco Plus")  # Integer
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
    # arrival_delay_in_minutes: int = Field(..., ge=0)
    satisfaction: int = Field(..., ge=0, le=1, description="0 for Neutral or Dissatisfied, 1 Satisfied")  # Integer

class Questions_passenger_satisfactionCreate(Questions_passenger_satisfaction_Base):
    pass

class Questions_passenger_satisfactionUpdate(BaseModel):
    gender: int | None = Field(None, ge=0, le=1)
    customer_type: int | None = Field(None, ge=0, le=1)
    age: int | None = Field(None, ge=0, le=120)
    travel_type: int | None = Field(None, ge=0, le=1)
    trip_class: int | None = Field(None, ge=0, le=2)
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
    satisfaction: int | None = Field(None, ge=0, le=1)

class Questions_passenger_satisfaction(Questions_passenger_satisfaction_Base):
    id: int

    class Config:
        orm_mode = True
