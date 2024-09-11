import streamlit as st
import requests
from enum import Enum

# URL de tu API FastAPI
API_URL = "http://localhost:8000"

# Definiciones de Enum para coincidir con el esquema
# class CustomerType(str, Enum):
#     LOYAL = "Loyal Customer"
#     DISLOYAL = "Disloyal Customer"

# class TravelType(str, Enum):
#     PERSONAL = "Personal Travel"
#     BUSINESS = "Business Travel"

# class TripClass(str, Enum):
#     ECO = "Eco"
#     ECO_PLUS = "Eco Plus"
#     BUSINESS = "Business"

# class Satisfaction(str, Enum):
#     NEUTRAL = "Neutral or Dissatisfied"
#     SATISFIED = "Satisfied"

# Configuración de la página
st.set_page_config(page_title="Satisfacción del Pasajero", page_icon="✈️")

page_bg_img = '''
<style>
.stApp {
background-image: url("https://www.aviationgroup.es/wp-content/uploads/2019/04/1132-e1591699339327.jpg);
background-size: cover;
}
</style>
'''

# Cargar el CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Título con emojis
st.title("📋 Formulario de Satisfacción del Pasajero 🛫😊")

st.write("¡Por favor, llena los siguientes campos para ayudarnos a mejorar! 🙏")

# --- Entrada de datos desde la interfaz ---

# Selección para el género
gender = st.selectbox("Seleccione Género:", ['Female', 'Male'])
input_data = {'gender': 0 if gender == 'Female' else 1}

# Selección para el tipo de cliente
customer_type = st.selectbox("Seleccione Tipo de Cliente:", ['Loyal Customer','Disloyal Customer'])
input_data = {'customer_type': 0 if customer_type == 'Disloyal Customer' else 1}

# Entrada para la edad (0 a 120)
age = st.slider("Seleccione Edad:", 0, 120, 25)
input_data['age'] = age

# Selección para el tipo de viaje
travel_type = st.selectbox("Seleccione Tipo de Viaje:", ['Personal Travel', 'Business Travel'])
input_data = {'travel_type': 0 if customer_type == 'Personal Travel' else 1}

# Selección para la clase
trip_class = st.selectbox("Seleccione Clase:", ['Eco', 'Eco Plus', 'Business'])
input_data = {
    'trip_class': 0 if trip_class == 'Business' 
                  else 1 if trip_class == 'Eco' 
                  else 2  
}

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

# # Entrada para el retraso de llegada (en minutos)
# arrival_delay = st.slider("Retraso en la Llegada (en minutos):", 0, 1000, 0)
# input_data['arrival_delay_in_minutes'] = arrival_delay

# Entrada para la satisfacción del cliente
satisfaction_client = st.selectbox("¿Está satisfecho con el servicio?", ['Neutral or Dissatisfied', 'Satisfied'])
input_data = {'satisfaction_client': 0 if satisfaction_client == 'Neutral or Dissatisfied' else 1}


# --- Función para enviar datos a la API y obtener predicción ---
def send_data_to_api(data):
    try:
        # Enviar datos para guardar y para predecir (ahora son los mismos)
        response_submit = requests.post(f"{API_URL}/submit/", json=data)
        response_submit.raise_for_status()
        
        response_predict = requests.post(f"{API_URL}/predict/", json=data)
        response_predict.raise_for_status()
        
        result_submit = response_submit.json()
        result_predict = response_predict.json()
        return result_submit, result_predict
    except requests.RequestException as e:
        st.error(f"Error al conectar con la API: {str(e)}")
        if e.response is not None:
            st.error(e.response.text)  # Mostrar respuesta completa del error
        return None, None

# --- Botón de enviar datos ---
if st.button("Guardar Datos y Predecir"):
    result_submit, result_predict = send_data_to_api(input_data)
    if result_submit and result_predict:
        st.success("Datos Guardados Exitosamente")
        prediction = result_predict["prediction"]
        st.success(f"La predicción de satisfacción del cliente es: {'Satisfecho' if prediction == 1 else 'No satisfecho'}")
    else:
        st.error("Hubo un error al procesar los datos. Por favor, intenta de nuevo.")

st.write("🎉 ¡Gracias por viajar con nosotros! ✈️💼 Esperamos que vueles pronto. 😊")