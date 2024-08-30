import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# Asumiendo que ya has cargado tu DataFrame como 'df'
# Si no, cárgalo así:
df = pd.read_csv('airline_passenger_satisfaction.csv')

# Identificar columnas categóricas
columnas_categoricas = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

# Inicializar el OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Ajustar y transformar las columnas categóricas
encoded_features = encoder.fit_transform(df[columnas_categoricas])

# Obtener los nombres de las nuevas columnas
feature_names = encoder.get_feature_names_out(columnas_categoricas)

# Crear un nuevo DataFrame con las características codificadas
encoded_df = pd.DataFrame(encoded_features, columns=feature_names)

# Concatenar el DataFrame original con las nuevas columnas codificadas
df_final = pd.concat([df.drop(columns=columnas_categoricas), encoded_df], axis=1)

# Mostrar las primeras filas del DataFrame resultante
print(df_final.head())

# Guardar el nuevo DataFrame si es necesario
# df_final.to_csv('datos_codificados.csv', index=False)