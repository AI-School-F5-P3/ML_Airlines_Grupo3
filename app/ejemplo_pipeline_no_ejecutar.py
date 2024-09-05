# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt

# Cargar el archivo CSV proporcionado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Separar características y etiquetas
X = df.drop(columns=['Arrival Delay in Minutes', 'satisfaction'])
y = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)  # Convertir a variable binaria si es necesario

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros para el modelo SVM
param_grid_svm = {
    'svm__C': [0.01, 0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'svm__gamma': ['scale', 'auto']
}

# Crear el pipeline con normalización y SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))  # Añadir probability=True para obtener probabilidades
])

# Configurar la búsqueda de hiperparámetros para el pipeline
grid_search_svm = GridSearchCV(estimator=pipeline, param_grid=param_grid_svm, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# Entrenar el modelo de SVM con la búsqueda de hiperparámetros
grid_search_svm.fit(X, y)

# Obtener el mejor modelo encontrado
best_svm_model = grid_search_svm.best_estimator_

# Evaluar el modelo usando validación cruzada para obtener predicciones de probabilidad
y_pred_proba = cross_val_predict(best_svm_model, X, y, cv=kf, method='predict_proba')[:, 1]
y_pred = cross_val_predict(best_svm_model, X, y, cv=kf, method='predict')

# Obtener la matriz de confusión
conf_matrix = confusion_matrix(y, y_pred)

# Calcular la curva ROC y el AUC
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = roc_auc_score(y, y_pred_proba)

# Mostrar resultados
print(f'Best parameters found for SVM: {grid_search_svm.best_params_}')
print(f'Cross-Validation Mean Accuracy: {np.mean(accuracy_score)}')
print(f'Cross-Validation Std Accuracy: {np.std(accuracy_score)}')
print(f'Cross-Validation Mean F1 Score: {np.mean(f1_score)}')
print(f'Cross-Validation Std F1 Score: {np.std(f1_score)}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'ROC AUC: {roc_auc}')

# Guardar los resultados en un DataFrame para comparación futura
results = pd.DataFrame({
    'Model': ['SVM'],
    'Best Parameters': [grid_search_svm.best_params_],
    'Test Accuracy': [np.mean(accuracy_score)],
    'Test F1 Score': [np.mean(f1_score)],
    'ROC AUC': [roc_auc]
})

# Guardar los resultados en un archivo CSV
results.to_csv('svm_classification_results.csv', index=False)
print("Results saved to 'svm_classification_results.csv'")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='SVM (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
