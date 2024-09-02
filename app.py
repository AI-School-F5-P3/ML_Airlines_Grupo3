import pandas as pd
import numpy as np
import joblib

# Cargar los objetos guardados
scaler = joblib.load('scaler.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
encoder = joblib.load('onehot_encoder.pkl')
model = joblib.load('best_decision_tree_model.pkl')

# Definir las características numéricas y categóricas como se usaron durante el entrenamiento
numeric_features = ["Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
                   "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
                   "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
                   "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
                   "Departure Delay in Minutes", "Arrival Delay in Minutes"]

categorical_features = ["Gender", "Customer Type", "Type of Travel"]
ordinal_features = ["Class"]

def collect_data_and_predict():
    print("Por favor, ingrese los siguientes datos (de 0 a 5 para los servicios y números para los demás campos):")

    # Datos numéricos del cliente
    gender = int(input("Gender (Male:1/Female:0): "))
    customer_type = int(input("Customer Type (Loyal Customer:0/Disloyal Customer:1): "))
    age = int(input("Age (0-120): "))
    type_of_travel = int(input("Type of Travel (Personal Travel:1/Business Travel:0): "))
    flight_class = int(input(("Class (Eco Plus:2/Business:0/Eco:1): ")))
    flight_distance = int(input("Flight Distance: "))
    inflight_wifi_service = int(input("Inflight wifi service (0-5): "))
    departure_arrival_time_convenient = int(input("Departure/Arrival time convenient (0-5): "))
    ease_of_online_booking = int(input("Ease of Online booking (0-5): "))
    gate_location = int(input("Gate location (0-5): "))
    food_and_drink = int(input("Food and drink (0-5): "))
    online_boarding = int(input("Online boarding (0-5): "))
    seat_comfort = int(input("Seat comfort (0-5): "))
    inflight_entertainment = int(input("Inflight entertainment (0-5): "))
    onboard_service = int(input("On-board service (0-5): "))
    leg_room_service = int(input("Leg room service (0-5): "))
    baggage_handling = int(input("Baggage handling (0-5): "))
    checkin_service = int(input("Checkin service (0-5): "))
    inflight_service = int(input("Inflight service (0-5): "))
    cleanliness = int(input("Cleanliness (0-5): "))
    departure_delay = int(input("Departure Delay in Minutes: "))
    arrival_delay = int(input("Arrival Delay in Minutes: "))
         
    # Crear DataFrame para las entradas del cliente
    client_data = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age': [age],
        'Type of Travel': [type_of_travel],
        'Class': [flight_class],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [inflight_wifi_service],              
        'Departure/Arrival time convenient': [departure_arrival_time_convenient],
        'Ease of Online booking': [ease_of_online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_and_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [inflight_entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room_service],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay]
    })

    # Escalar los datos del cliente utilizando el mismo escalador que para los datos de entrenamiento
    client_data_scaled = scaler.transform(client_data[numeric_features])

    # Codificar las variables categóricas
    client_categorical = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Type of Travel': [type_of_travel],
        'Class': [flight_class]
    })
    
    client_categorical_encoded = pd.DataFrame(encoder.transform(client_categorical[['Gender', 'Customer Type', 'Type of Travel']]).toarray(), 
                                              columns=encoder.get_feature_names_out(['Gender', 'Customer Type', 'Type of Travel']))
    client_categorical_encoded['Class'] = ordinal_encoder.transform(client_categorical[['Class']])
    
    # Concatenar las características codificadas con las características numéricas
    client_data_processed = pd.concat([pd.DataFrame(client_data_scaled, columns=numeric_features), client_categorical_encoded], axis=1)
    
    # Realizar la predicción
    prediction = model.predict(client_data_processed)
    prediction_proba = model.predict_proba(client_data_processed)[:, 1]

    # Mostrar los resultados
    if prediction[0] == 1:
        print("El cliente estará satisfecho.")
    else:
        print("El cliente no estará satisfecho.")
    print(f"Probabilidad de satisfacción: {prediction_proba[0]:.2f}")

# Ejecutar la función para recolectar datos y predecir
collect_data_and_predict()
