import streamlit as st
import requests

# Diccionarios de mapeo para campos categóricos
customer_type_map = {'Loyal Customer': 'loyal', 'Disloyal Customer': 'disloyal'}
travel_type_map = {'Personal Travel': 'personal', 'Business Travel': 'business'}
class_map = {'Eco': 'eco', 'Eco Plus': 'eco_plus', 'Business': 'business'}
satisfaction_map = {'Neutral or Dissatisfied': 'neutral', 'Satisfied': 'satisfied'}

# Título con emojis
st.title("📋 Formulario de Satisfacción del Pasajero 🛫😊")

# Resto del código para tu formulario
st.write("¡Por favor, llena los siguientes campos para ayudarnos a mejorar! 🙏")

# --- Entrada de datos desde la interfaz ---

# Selección para el género (0 para Female, 1 para Male)
gender = st.selectbox("Seleccione Género:", ['Female', 'Male'])
input_data = {'gender': 0 if gender == 'Female' else 1}

# Selección para el tipo de cliente
customer_type = st.selectbox("Seleccione Tipo de Cliente:", ['Loyal Customer', 'Disloyal Customer'])
input_data['customer_type'] = customer_type_map[customer_type]

# Entrada para la edad (0 a 120)
age = st.slider("Seleccione Edad:", 0, 120, 25)
input_data['age'] = age

# Selección para el tipo de viaje
travel_type = st.selectbox("Seleccione Tipo de Viaje:", ['Personal Travel', 'Business Travel'])
input_data['travel_type'] = travel_type_map[travel_type]

# Selección para la clase
trip_class = st.selectbox("Seleccione Clase:", ['Eco', 'Eco Plus', 'Business'])
input_data['trip_class'] = class_map[trip_class]

# Entrada para la distancia de vuelo (0 a 10000)
flight_distance = st.slider("Distancia de Vuelo (km):", 0, 10000, 500)
input_data['flight_distance'] = flight_distance

# Preguntas de satisfacción (escala de 0 a 5)
def get_satisfaction(label: str):
    return st.slider(label, 0, 5, 3)

input_data['inflight_wifi_service'] = get_satisfaction("Servicio de Wifi a Bordo")
input_data['departure_arrival_time_convenient'] = get_satisfaction("Conveniencia del Tiempo de Salida/Llegada")
input_data['online_booking'] = get_satisfaction("Facilidad de Reserva Online")
input_data['gate_location'] = get_satisfaction("Ubicación de la Puerta")
input_data['food_and_drink'] = get_satisfaction("Comida y Bebida")
input_data['online_boarding'] = get_satisfaction("Embarque Online")
input_data['seat_comfort'] = get_satisfaction("Comodidad del Asiento")
input_data['inflight_entertainment'] = get_satisfaction("Entretenimiento a Bordo")
input_data['onboard_service'] = get_satisfaction("Servicio a Bordo")
input_data['leg_room_service'] = get_satisfaction("Espacio para las Piernas")
input_data['baggage_handling'] = get_satisfaction("Manejo del Equipaje")
input_data['checkin_service'] = get_satisfaction("Servicio de Check-in")
input_data['inflight_service'] = get_satisfaction("Servicio en Vuelo")
input_data['cleanliness'] = get_satisfaction("Limpieza")

# Entrada para el retraso de salida (en minutos)
departure_delay = st.slider("Retraso en la Salida (en minutos):", 0, 1000, 0)
input_data['departure_delay_in_minutes'] = departure_delay

# Entrada para el retraso de llegada (en minutos)
arrival_delay = st.slider("Retraso en la Llegada (en minutos):", 0, 1000, 0)
input_data['arrival_delay_in_minutes'] = arrival_delay

# Entrada para la satisfacción del cliente
satisfaction_client = st.selectbox("¿Está satisfecho con el servicio?", ['Neutral or Dissatisfied', 'Satisfied'])
input_data['satisfaction'] = satisfaction_map[satisfaction_client]

# --- Función para enviar datos a la API ---
def send_data_to_api(data):
    try:
        response = requests.post("http://localhost:8000/submit/", json=data)
        response.raise_for_status()
        result = response.json()
        return result
    except requests.RequestException as e:
        st.error(f"Error al conectar con la API: {str(e)}")
        if e.response is not None:
            st.error(e.response.text)  # Mostrar respuesta completa del error
        return None

st.write("🎉 ¡Gracias por viajar con nosotros! ✈️💼 Esperamos que vueles pronto. 😊")


# --- Botón de enviar datos ---
if st.button("Guardar Datos"):
    st.write(input_data)  # Muestra los datos ingresados en la interfaz para verificación
    result = send_data_to_api(input_data)
    message = "Datos Guardados"
    if result:
        if 'error' in result:
            st.error(result['error'])
        else:
            st.success(result['message'])


