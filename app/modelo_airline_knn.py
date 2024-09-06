import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = 'data/airline_modified_knn.csv'
df = pd.read_csv(file_path)

# Manejar valores faltantes
df = df.fillna(df.median())

# Dividir el dataset
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir el modelo KNN y los hiperparámetros
knn_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Búsqueda de hiperparámetros con GridSearchCV
grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_knn_model = grid_search.best_estimator_

# Evaluar el modelo con validación cruzada
cv_scores = cross_val_score(best_knn_model, X_train, y_train, cv=5, scoring='accuracy')
y_pred = cross_val_predict(best_knn_model, X_test, y_test, cv=5)
y_pred_proba = cross_val_predict(best_knn_model, X_test, y_test, cv=5, method='predict_proba')[:, 1]

# Métricas con pos_label especificado
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Mostrar resultados
print(f'Mejores parámetros KNN: {grid_search.best_params_}')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Guardar el modelo
joblib.dump(best_knn_model, 'models/knn_model.pkl')

# Guardar resultados en un CSV
results = pd.DataFrame({
    'Model': ['KNN'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'ROC AUC': [roc_auc],
    'Best Parameters': [grid_search.best_params_],
})
results.to_csv('knn_classification_results.csv', index=False)

# Guardar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'KNN (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC KNN')
plt.legend(loc="lower right")
plt.savefig('roc_curve_knn.png')
plt.close()
