import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Cargar el archivo CSV proporcionado
file_path = 'Airlines_Modified.csv'
df = pd.read_csv(file_path)

# # Preprocesamiento
# df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
# df.dropna(subset=['satisfaction'], inplace=True)

# Codificación de variables categóricas
label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Dividimos el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = df['satisfaction']

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializamos y entrenamos el modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Función para solicitar datos al usuario y predecir la satisfacción
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
   
       

    # # Codificación de datos categóricos
    # gender_encoded = label_encoder.transform([gender])[0]
    # customer_type_encoded = label_encoder.transform([customer_type])[0]
    # type_of_travel_encoded = label_encoder.transform([type_of_travel])[0]
    # flight_class_encoded = label_encoder.transform([flight_class])[0]

    # Crear DataFrame para las entradas del cliente
    client_data = pd.DataFrame({
        'Gender': [gender],
        'Customer Type': [customer_type],
        'Age':[age],
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

    # Verificar los datos del cliente
    print("Datos del cliente para predicción:")
    print(client_data)

    # Verificar si hay valores NaN en client_data
    if client_data.isna().sum().sum() > 0:
        print("Error: Algunos datos contienen valores NaN. Por favor, verifique las entradas.")
        return

    # Realizar la predicción
    prediction = rf_model.predict(client_data)
    prediction_proba = rf_model.predict_proba(client_data)[:, 1]

    # Mostrar los resultados
    if prediction[0] == 1:
        print("El cliente estará satisfecho.")
    else:
        print("El cliente no estará satisfecho.")
    print(f"Probabilidad de satisfacción: {prediction_proba[0]:.2f}")

# Ejecutar la función para recolectar datos y predecir
collect_data_and_predict()
