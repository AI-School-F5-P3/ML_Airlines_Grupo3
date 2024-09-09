import streamlit as st
import requests

# Diccionarios de mapeo para campos categóricos
gender_map = {0: 'Female', 1: 'Male'}
customer_type_map = {'Loyal': 'loyal', 'Disloyal': 'disloyal'}
travel_type_map = {'Personal Travel': 'personal', 'Business Travel': 'business'}
class_map = {'Eco': 'eco', 'Eco Plus': 'eco plus', 'Business': 'business'}
satisfaction_map = {'Neutral or Dissatisfied': 'neutral', 'Satisfied': 'satisfied'}

# Crear la interfaz de Streamlit
st.title("Formulario de Satisfacción del Pasajero")

# Selección para el género
gender = st.selectbox("Seleccione Género:", ['Female', 'Male'], key="gender")
input_data = {'gender': list(gender_map.keys())[list(gender_map.values()).index(gender)]}

# Selección para el tipo de cliente
customer_type = st.selectbox("Seleccione Tipo de Cliente:", ['Loyal', 'Disloyal'], key="customer_type")
input_data['customer_type'] = customer_type_map[customer_type]

# Entrada para la edad (0 a 120)
age = st.slider("Seleccione Edad:", 0, 120, 25, key="age")
input_data['age'] = age

# Selección para el tipo de viaje
travel_type = st.selectbox("Seleccione Tipo de Viaje:", ['Personal Travel', 'Business Travel'], key="travel_type")
input_data['travel_type'] = travel_type_map[travel_type]

# Selección para la clase
trip_class = st.selectbox("Seleccione Clase:", ['Eco', 'Eco Plus', 'Business'], key="trip_class")
input_data['trip_class'] = class_map[trip_class]

# Entrada para la distancia de vuelo (0 a 10000)
flight_distance = st.slider("Distancia de Vuelo (km):", 0, 10000, 500, key="flight_distance")
input_data['flight_distance'] = flight_distance

# Preguntas de satisfacción (escala de 0 a 5)
def get_satisfaction(label: str):
    return st.slider(label, 0, 5, 3, key=label)

inflight_wifi = get_satisfaction("Servicio de Wifi a Bordo")
input_data['inflight_wifi_service'] = inflight_wifi

departure_arrival_time_convenient = get_satisfaction("Conveniencia del Tiempo de Salida/Llegada")
input_data['departure_arrival_time_convenient'] = departure_arrival_time_convenient

online_booking = get_satisfaction("Facilidad de Reserva Online")
input_data['online_booking'] = online_booking

gate_location = get_satisfaction("Ubicación de la Puerta")
input_data['gate_location'] = gate_location

food_and_drink = get_satisfaction("Comida y Bebida")
input_data['food_and_drink'] = food_and_drink

online_boarding = get_satisfaction("Embarque Online")
input_data['online_boarding'] = online_boarding

seat_comfort = get_satisfaction("Comodidad del Asiento")
input_data['seat_comfort'] = seat_comfort

inflight_entertainment = get_satisfaction("Entretenimiento a Bordo")
input_data['inflight_entertainment'] = inflight_entertainment

onboard_service = get_satisfaction("Servicio a Bordo")
input_data['onboard_service'] = onboard_service

leg_room_service = get_satisfaction("Espacio para las Piernas")
input_data['leg_room_service'] = leg_room_service

baggage_handling = get_satisfaction("Manejo del Equipaje")
input_data['baggage_handling'] = baggage_handling

checkin_service = get_satisfaction("Servicio de Check-in")
input_data['checkin_service'] = checkin_service

inflight_service = get_satisfaction("Servicio en Vuelo")
input_data['inflight_service'] = inflight_service

cleanliness = get_satisfaction("Limpieza")
input_data['cleanliness'] = cleanliness

# Entrada para el retraso de salida (en minutos)
departure_delay = st.slider("Retraso en la Salida (en minutos):", 0, 1000, 0, key="departure_delay")
input_data['departure_delay_in_minutes'] = departure_delay

# Entrada para el retraso de llegada (en minutos)
arrival_delay = st.slider("Retraso en la Llegada (en minutos):", 0, 1000, 0, key="arrival_delay")
input_data['arrival_delay_in_minutes'] = arrival_delay

# Entrada para la satisfacción del cliente
satisfaction_client = st.selectbox("¿Está satisfecho con el servicio?", ['Neutral or Dissatisfied', 'Satisfied'], key="satisfaction_client")
input_data['satisfaction'] = satisfaction_map[satisfaction_client]

# Función para enviar datos a la API FastAPI
def send_data_to_api(data):
    try:
        response = requests.post("http://localhost:8000/submit/", json=data)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.RequestException as e:
        st.error(f"Error al conectar con la API: {str(e)}")
        return None

# Botón de guardar datos
if st.button("Guardar Datos"):
    st.write(input_data)  # Muestra los datos ingresados en la interfaz
    result = send_data_to_api(input_data)
    if result:
        if 'error' in result:
            st.error(result['error'])
        else:
            st.success(result['message'])
