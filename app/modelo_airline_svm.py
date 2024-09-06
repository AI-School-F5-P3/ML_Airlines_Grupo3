import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils import shuffle
import time
import joblib
import matplotlib.pyplot as plt

# Cargar el archivo CSV proporcionado
print("Cargando datos...")
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Usar una muestra para todo el proceso
sample_size = 50000
df_sample = shuffle(df, random_state=42).iloc[:sample_size]

X = df_sample.drop(columns=['Arrival Delay in Minutes', 'satisfaction'])
y = df_sample['satisfaction']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definir el espacio de búsqueda de hiperparámetros para SVM lineal
param_grid_svm = {
    'C': [0.1, 1, 10],
}

# Configurar la búsqueda de hiperparámetros para SVM lineal
svm_model = LinearSVC(random_state=42, max_iter=2000)

# Realizar la búsqueda de hiperparámetros
print("Iniciando búsqueda de hiperparámetros...")
start_time = time.time()
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search_svm.fit(X_train, y_train)
end_time = time.time()

print(f"Búsqueda de hiperparámetros completada en {end_time - start_time:.2f} segundos")

# Evaluar el modelo ajustado en el conjunto de test
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)

# Obtener el mejor modelo
best_svm_model = grid_search_svm.best_estimator_

# Entrenar el mejor modelo con TODO el conjunto de datos
print("Entrenando el mejor modelo con todo el dataset...")
best_svm_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = best_svm_model.predict(X_test)

# Calcular métricas
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Para ROC AUC, usamos las puntuaciones de decisión
y_score = best_svm_model.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = roc_auc_score(y_test, y_score)

# Imprimir los resultados
print(f"La exactitud del modelo SVM es de {accuracy * 100:.2f}%")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# ... (resto del código sin cambios)

# Dibujar y guardar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC (Linear SVM)')
plt.legend(loc="lower right")
plt.savefig('roc_curve_svm.png')
plt.close()
print("Curva ROC guardada como 'roc_curve_svm.png'")