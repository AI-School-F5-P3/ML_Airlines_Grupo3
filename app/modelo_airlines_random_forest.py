# Importamos las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib


# Cargar el archivo CSV proporcionado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)


# Dividimos el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Dividimos los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializamos y entrenamos el modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test)

# Evaluamos el rendimiento del modelo de Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Random Forest AUC-ROC Score:", roc_auc_score(y_test, y_pred_rf))

#Verificación adicional sobre la precisión del modelo
accuracy = accuracy_score(y_test,y_pred_rf)
print(f"La exactitud del modelo de clasificación es de {accuracy * 100:.2f}%")

# Guardar el modelo en un archivo
joblib.dump(rf_model, 'models/rf_model.pkl')
print("Modelo guardado como rf_model.pkl")