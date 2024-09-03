import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# Cargar el archivo CSV proporcionado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Dividimos el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Escalamos los datos para normalizar las características
escalador = preprocessing.MinMaxScaler()
X_scaled = escalador.fit_transform(X)

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Inicializamos el modelo de K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)  # Ajusta el número de vecinos según sea necesario

# Validación cruzada (cross-validation) con 5 "folds"
cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")

# Entrenamos el modelo KNN en los datos de entrenamiento
knn_model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred_knn = knn_model.predict(X_test)

# Evaluamos el rendimiento del modelo de K-Nearest Neighbors (KNN)
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("KNN AUC-ROC Score:", roc_auc_score(y_test, y_pred_knn))

# Verificación adicional sobre la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"La exactitud del modelo KNN es de {accuracy * 100:.2f}%")

# Verificación de overfitting comparando la precisión de entrenamiento con la precisión de validación
train_accuracy = knn_model.score(X_train, y_train)
test_accuracy = knn_model.score(X_test, y_test)

print(f"\nTrain Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if abs(train_accuracy - test_accuracy) < 0.05:
    print("No hay indicios significativos de overfitting (diferencia de precisión < 5%).")
else:
    print("Posible overfitting detectado (diferencia de precisión > 5%).")
