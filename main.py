from fastapi import FastAPI, Form
from pydantic import BaseModel


app = FastAPI()

class Questions_passenger_satisfaction(BaseModel):
    id: int
    gender: bool
    customer_type: bool
    age: int
    travel_type: bool
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
    
@app.post("/submit/")
async def submit_form(
    id: int = Form(...),
    gender: bool = Form(...),
    customer_type: bool = Form(...),
    age: int = Form(...),
    travel_type: bool = Form(...),
    trip_class: str = Form(...),
    flight_distance: int = Form(...),
    inflight_wifi_service: int = Form(...),
    departure_arrival_time_convenient: int = Form(...),
    online_booking: int = Form(...),
    gate_location: int = Form(...),
    food_and_drink: int = Form(...),
    online_boarding: int = Form(...),
    seat_comfort: int = Form(...),
    inflight_entertainment: int = Form(...),
    onboard_service: int = Form(...),
    leg_room_service: int = Form(...),
    baggage_handling: int = Form(...),
    checkin_service: int = Form(...),
    inflight_service: int = Form(...),
    cleanliness: int = Form(...),
    departure_delay_in_minutes: int = Form(...),
    arrival_delay_in_minutes: int = Form(...),
    passenger_satisfaction: str = Form(...)
):
    data = Questions_passenger_satisfaction(
        gender=gender,
        customer_type=customer_type,
        age=age,
        travel_type=travel_type,
        trip_class=trip_class,
        flight_distance=flight_distance,
        inflight_wifi_service=inflight_wifi_service,
        departure_arrival_time_convenient=departure_arrival_time_convenient,
        online_booking=online_booking,
        gate_location=gate_location,
        food_and_drink=food_and_drink,
        online_boarding=online_boarding,
        seat_comfort=seat_comfort,
        inflight_entertainment=inflight_entertainment,
        onboard_service=onboard_service,
        leg_room_service=leg_room_service,
        baggage_handling=baggage_handling,
        checkin_service=checkin_service,
        inflight_service=inflight_service,
        cleanliness=cleanliness,
        departure_delay_in_minutes=departure_delay_in_minutes,
        arrival_delay_in_minutes=arrival_delay_in_minutes,
        passenger_satisfaction=passenger_satisfaction
    )
    return {"message": "Datos recibidos correctamente", "data": data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    


        
