# Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score


# Cargar el archivo CSV proporcionado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Dividimos el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializamos y entrenamos el modelo de K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos (n_neighbors) según sea necesario
knn_model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred_knn = knn_model.predict(X_test)

# Evaluamos el rendimiento del modelo de K-Nearest Neighbors (KNN)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("KNN AUC-ROC Score:", roc_auc_score(y_test, y_pred_knn))

# Verificación adicional sobre la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"La exactitud del modelo KNN es de {accuracy * 100:.2f}%")
