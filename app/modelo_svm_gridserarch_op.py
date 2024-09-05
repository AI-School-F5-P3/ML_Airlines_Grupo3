import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

# Cargar y preparar los datos
print("Cargando datos...")
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Usar una muestra de 10.000 filas
sample_size = 10000
df_sample = shuffle(df, random_state=42).iloc[:sample_size]

X = df_sample.drop(columns=['Arrival Delay in Minutes', 'satisfaction'])
y = df_sample['satisfaction']

# Configurar la búsqueda de hiperparámetros
param_grid_svm = {
    'C': [1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

# Configurar el modelo SVM
svm_model = SVC(probability=True, random_state=42)

# Realizar la búsqueda de hiperparámetros
print("Iniciando búsqueda de hiperparámetros...")
start_time = time.time()
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_svm.fit(X, y)
end_time = time.time()

print(f"Búsqueda de hiperparámetros completada en {end_time - start_time:.2f} segundos")

# Obtener el mejor modelo
best_svm_model = grid_search_svm.best_estimator_

# Realizar validación cruzada para la evaluación final
print("Realizando validación cruzada final...")
cv_scores = cross_val_score(best_svm_model, X, y, cv=5, scoring='accuracy')
y_pred = cross_val_predict(best_svm_model, X, y, cv=5)
y_pred_proba = cross_val_predict(best_svm_model, X, y, cv=5, method='predict_proba')[:, 1]

# Calcular métricas
conf_matrix = confusion_matrix(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = roc_auc_score(y, y_pred_proba)
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred, average='weighted')

# Mostrar resultados
print(f'Mejores parámetros encontrados para SVM: {grid_search_svm.best_params_}')
print(f'Precisión media de validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
print(f'Precisión: {accuracy:.4f}')
print(f'Puntuación F1: {f1:.4f}')
print(f'Matriz de confusión:\n{conf_matrix}')
print(f'ROC AUC: {roc_auc:.4f}')

# Guardar resultados
results = pd.DataFrame({
    'Model': ['SVM (10000 muestras)'],
    'Best Parameters': [grid_search_svm.best_params_],
    'CV Accuracy Mean': [cv_scores.mean()],
    'CV Accuracy Std': [cv_scores.std()],
    'Test Accuracy': [accuracy],
    'Test F1 Score': [f1],
    'ROC AUC': [roc_auc]
})
results.to_csv('svm_classification_results_10000_samples.csv', index=False)
print("Resultados guardados en 'svm_classification_results_10000_samples.csv'")

# Dibujar y guardar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'SVM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC (10000 muestras)')
plt.legend(loc="lower right")
plt.savefig('roc_curve_svm_10000_samples.png')
plt.close()
print("Curva ROC guardada como 'roc_curve_svm_10000_samples.png'")