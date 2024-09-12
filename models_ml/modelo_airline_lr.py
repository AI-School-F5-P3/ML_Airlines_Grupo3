import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
import joblib

# Cargar y preparar los datos
print("Cargando datos...")
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Usar una muestra de 10.000 filas para la búsqueda de hiperparámetros
sample_size = 10000
df_sample = shuffle(df, random_state=42).iloc[:sample_size]

X_sample = df_sample.drop(columns=['Arrival Delay in Minutes', 'satisfaction'])
y_sample = df_sample['satisfaction']

# Preparar todo el dataset para el entrenamiento final
X_full = df.drop(columns=['Arrival Delay in Minutes', 'satisfaction'])
y_full = df['satisfaction']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Configurar la búsqueda de hiperparámetros
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Configurar el modelo de Regresión Logística
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Realizar la búsqueda de hiperparámetros con la muestra
print("Iniciando búsqueda de hiperparámetros...")
start_time = time.time()
grid_search_lr = GridSearchCV(estimator=lr_model, param_grid=param_grid_lr, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_lr.fit(X_sample, y_sample)
end_time = time.time()

print(f"Búsqueda de hiperparámetros completada en {end_time - start_time:.2f} segundos")

# Obtener el mejor modelo
best_lr_model = grid_search_lr.best_estimator_

# Entrenar el mejor modelo con TODO el conjunto de datos
print("Entrenando el mejor modelo con todo el dataset...")
best_lr_model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = best_lr_model.predict(X_test)
y_pred_proba = best_lr_model.predict_proba(X_test)[:, 1]

# Calcular métricas
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Mostrar resultados
print(f'Mejores parámetros encontrados para Logistic Regression: {grid_search_lr.best_params_}')
print(f'Precisión: {accuracy:.4f}')
print(f'Precisión (Precision): {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Puntuación F1: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'Matriz de confusión:\n{conf_matrix}')

# Guardar el modelo en un archivo
joblib.dump(best_lr_model, 'models/lr_model.pkl')
print("Modelo guardado como lr_model.pkl")

# Guardar resultados
results = pd.DataFrame({
    'Model': ['Logistic Regression'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'ROC AUC': [roc_auc],
    'Best Parameters': [grid_search_lr.best_params_],
})

# Actualizar el archivo CSV de métricas
try:
    existing_metrics = pd.read_csv('metrics/model_metrics.csv')
    updated_metrics = pd.concat([existing_metrics, results], ignore_index=True)
except FileNotFoundError:
    updated_metrics = results

updated_metrics.to_csv('metrics/model_metrics.csv', index=False)
print("Métricas guardadas en 'model_metrics.csv'")

# Dibujar y guardar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC (Logistic Regression)')
plt.legend(loc="lower right")
plt.savefig('roc_curve_lr.png')
plt.close()
print("Curva ROC guardada como 'roc_curve_lr.png'")