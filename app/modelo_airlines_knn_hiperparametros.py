# Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib

# Cargar el archivo CSV proporcionado
file_path = 'data/airline_modified_knn.csv'
df = pd.read_csv(file_path)

# Verificar si hay valores NaN o infinitos en el dataset
print("Valores faltantes por columna:")
print(df.isnull().sum())

print("Hay valores infinitos en el dataset?:", np.isinf(df).values.any())

# Si hay valores faltantes (NaN), podrías llenarlos o eliminarlos
# Ejemplo: rellenar con la mediana (o podrías usar la media o eliminar filas)
df = df.fillna(df.median())

# Alternativamente, eliminar filas con valores faltantes
# df = df.dropna()

# Dividimos el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de datos
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definimos el modelo base de K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier()

# Definimos el rango de hiperparámetros para la búsqueda
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Diferentes valores para n_neighbors
    'weights': ['uniform', 'distance'],  # Peso uniforme o basado en distancia
    'metric': ['euclidean', 'manhattan']  # Métricas de distancia
}

# Configuramos la búsqueda de hiperparámetros usando GridSearchCV
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Entrenamos el modelo usando GridSearchCV
grid_search.fit(X_train, y_train)

# Mostramos los mejores hiperparámetros encontrados por GridSearchCV
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Usamos el mejor modelo encontrado para hacer predicciones
best_knn_model = grid_search.best_estimator_
y_pred_best_knn = best_knn_model.predict(X_test)

# Evaluamos el rendimiento del mejor modelo de K-Nearest Neighbors (KNN)
print("\nKNN Classification Report (Mejor Modelo):")
print(classification_report(y_test, y_pred_best_knn))
print("KNN Confusion Matrix (Mejor Modelo):")
print(confusion_matrix(y_test, y_pred_best_knn))
print("KNN AUC-ROC Score (Mejor Modelo):", roc_auc_score(y_test, y_pred_best_knn))

# Verificación adicional sobre la precisión del mejor modelo
best_accuracy = accuracy_score(y_test, y_pred_best_knn)
print(f"La exactitud del mejor modelo KNN es de {best_accuracy * 100:.2f}%")

# Guardar el modelo en un archivo
joblib.dump(best_knn_model, 'models/knn_model.pkl')
print("Modelo guardado como knn_model.pkl")
