import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import seaborn as sns

# Cargar el archivo CSV proporcionado
file_path = 'data/airline_modified_knn.csv'
df = pd.read_csv(file_path)

# Verificar si hay valores NaN o infinitos en el dataset
print("Valores faltantes por columna:")
print(df.isnull().sum())

print("¿Hay valores infinitos en el dataset?:", np.isinf(df).values.any())

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

# Realizar validación cruzada con el mejor modelo
cv_scores = cross_val_score(best_knn_model, X_train, y_train, cv=5, scoring='accuracy')
print("\nResultados de la Validación Cruzada:")
print(f"Exactitudes por fold: {cv_scores}")
print(f"Exactitud media en la validación cruzada: {np.mean(cv_scores) * 100:.2f}%")
print(f"Desviación estándar de la exactitud: {np.std(cv_scores) * 100:.2f}%")

# Cálculo de métricas adicionales
accuracy = accuracy_score(y_test, y_pred_best_knn)
precision = precision_score(y_test, y_pred_best_knn, average='binary')  # Cambia 'binary' según el tipo de clasificación
recall = recall_score(y_test, y_pred_best_knn, average='binary')
f1 = f1_score(y_test, y_pred_best_knn, average='binary')
roc_auc= roc_auc_score(y_test, y_pred_best_knn)


# Guardar el modelo en un archivo
joblib.dump(best_knn_model, 'models/knn_model.pkl')
print("Modelo guardado como knn_model.pkl")


#Métricas
metricsdf = pd.DataFrame({
    'Model': ['KNN'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1],
    'AUC_ROC': [roc_auc],
    'Best_Parameters': [str(grid_search.best_params_)]
})

#Carga de df
try:
    existing_metrics = pd.read_csv('model_metrics.csv')
    updated_metrics = pd.concat([existing_metrics, metricsdf], ignore_index=True)
except FileNotFoundError:
    updated_metrics = metricsdf

updated_metrics.to_csv('model_metrics.csv', index=False)
print("Métricas guardadas en 'model_metrics.csv'")

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC'], 
            y=[accuracy, precision, recall, f1, roc_auc])
plt.title('Métricas del Modelo KNN Vecinos Cercanos')
plt.ylim(0, 1)
plt.savefig('knn_metrics.png')
plt.close()
print("Gráfico de métricas guardado como 'knn_metrics.png'")