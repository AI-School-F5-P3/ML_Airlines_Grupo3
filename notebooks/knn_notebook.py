import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

"""Fuente: https://www.youtube.com/watch?v=XN6fChNqfbs"""

#Datos de pasajeros

pasajeros = pd.read_csv("data/Airlines_Modified_knn.csv")
pasajeros

#Datos de pasajeros satisfechos y neutrales o No Satisfechos

satisfechos = pasajeros[pasajeros["satisfaction"]==1]
no_satisfechos = pasajeros[pasajeros["satisfaction"]==0]
print(satisfechos)
print(no_satisfechos)

#Gráfica Satisfechos y No satisfechos por Edad y por Clase

plt.scatter(satisfechos["Age"], satisfechos["Class"],
            marker="*", s=10, color="skyblue",
            label="Satisfecho")

plt.scatter(no_satisfechos["Age"], no_satisfechos["Class"],
            marker="*", s=10, color="red", 
            label="No Satisfecho")

plt.ylabel("Grado de Satisfacción de pasajeros por Edad y Clase")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2)) 
plt.show()

#Gráfica Satisfechos y No satisfechos por Edad y por Distancia de Vuelo

plt.scatter(satisfechos["Age"], satisfechos["Flight Distance"],
            marker="*", s=10, color="skyblue",
            label="Satisfecho")

plt.scatter(no_satisfechos["Age"], no_satisfechos["Flight Distance"],
            marker="*", s=10, color="red", 
            label="No Satisfecho")

plt.ylabel("Grado de Satisfacción de pasajeros por Edad y por Distancia de Vuelo")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2)) 
plt.show()

#Escalado de datos
"""Números mas pequeños = 0, números más grandes= 1"""

datos = pasajeros[['Gender', 'Customer Type', 'Age', 'Type of Travel',
       'Class', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']]
clase = pasajeros["satisfaction"]

escalador = preprocessing.MinMaxScaler()

datos = escalador.fit_transform(datos)

#Creación del modelo KNN
"""El número de vecinos que se utiliza para crear este tipo de modelo puede variar. Por. Ejemplo la raíz cuadrada del número de
casos"""

clasificador = KNeighborsClassifier(n_neighbors=3)

clasificador.fit(datos, clase) #con el método fit se utiliza el método clasificador

#Nuevo Pasajero

gender = 1
customer_type = 0
age = 26
type_of_travel = 0
flight_class = 0
flight_distance = 712
inflight_wifi_service = 4
departure_arrival_time_convenient = 4
ease_of_online_booking = 4
gate_location = 4
food_and_drink = 5
online_boarding = 5
seat_comfort = 5
inflight_entertainment = 5
onboard_service = 3
leg_room_service = 4
baggage_handling = 4
checkin_service = 3
inflight_service = 4
cleanliness = 5
departure_delay = 17
arrival_delay = 26

pasajero_data = pd.DataFrame({
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


#Escalar los datos del nuevo solicitante
nuevo_pasajero = escalador.transform(pasajero_data)

#Calcular clase y probabilidades
print("Clase:", clasificador.predict(nuevo_pasajero))
print("Probabilidades por clase",
      clasificador.predict_proba(nuevo_pasajero))

#Código para graficar
plt.scatter(satisfechos["Age"], satisfechos["Flight Distance"],
            marker="*", s=10, color="skyblue", label="Satisfecho")
plt.scatter(no_satisfechos["Age"], no_satisfechos["Flight Distance"],
            marker="*", s=10, color="red", label="Neutral o Insatisfecho")
plt.scatter(age, flight_distance, marker="P", s=250, color="green", label="Nuevo Pasajero") 
plt.ylabel("Grado de Satisfacción Hipotética de Nuevo Pasajero")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.3))
plt.show()

# #Regiones de las clases

# #Datos sinténticos de todos los posibles solicitantes
# class = np.array([np.arange(0, 1, 2)]*43).reshape(1, -1)
# age = np.array([np.arange(18, 61)]*501).reshape(1, -1)
# pasajeros = pd.DataFrame(np.stack((age, class), axis=2)[0],
#                      columns=["edad", "credito"])

# #Escalar los datos
# solicitantes = escalador.transform(todos)

# #Predecir todas las clases
# clases_resultantes = clasificador.predict(solicitantes)

# #Código para graficar
# buenos = todos[clases_resultantes==1]
# malos = todos[clases_resultantes==0]
# plt.scatter(buenos["age"], buenos["class"],
#             marker="*", s=150, color="skyblue", label="Estará satisfecho")
# plt.scatter(malos["age"], malos["class"],
#             marker="*", s=150, color="red", label="No estará satisfecho o será neutral")
# plt.ylabel("Monto del crédito")
# plt.xlabel("Edad")
# plt.legend(bbox_to_anchor=(1, 0.2))
# plt.show()