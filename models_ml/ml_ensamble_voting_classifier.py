import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import joblib
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# Cargar el archivo CSV proporcionado - Dataset limpio sin escalar.
file_path = 'data/airline_clean_data.csv'
df = pd.read_csv(file_path)


# Dividimos el dataset en características (X) y variable objetivo (y)
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos (es necesario para Logistic Regression y KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir los modelos individuales
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
logistic_regression = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Crear el ensemble usando VotingClassifier
# 'soft' usa las probabilidades de los modelos, mientras que 'hard' usa las predicciones finales
ensemble_model = VotingClassifier(estimators=[
    ('rf', random_forest),
    ('lr', logistic_regression),
    ('knn', knn)
], voting='soft')  # Para votar basado en las probabilidades (mejor si los modelos lo permiten)

# Entrenar el ensemble
ensemble_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = ensemble_model.predict(X_test)
y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]  # Usar solo la probabilidad de la clase positiva

# Evaluar el rendimiento del ensemble
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # Aquí ahora se usa solo la probabilidad de la clase positiva
roc_auc = roc_auc_score(y_test, y_pred_proba)  # También para el AUC se usa solo la probabilidad de la clase positiva
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)


print(f"Accuracy del modelo ensemble: {accuracy}")
print(f"Matriz de confusión:\n{conf_matrix}")
print(f"Reporte de clasificación:\n{classification_report(y_test, y_pred)}")
print("Random Forest AUC:", roc_auc)


precision = precision_score(y_test, y_pred, average='binary')  # Cambia 'binary' según el tipo de clasificación
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Guardar el modelo en un archivo
joblib.dump(ensemble_model, 'models_ml/pkls/ensemble_model.pkl')
print("Modelo guardado como ensemble_model.pkl")

# Métricas
metricsdf = pd.DataFrame({
    'Model': ['EM'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'ROC AUC': [roc_auc],
    'Best Parameters': [str("Hiperparámetros No Aplicados en este modelo")]
})




# Cargar métricas existentes (si las hay) y guardar en un archivo CSV
try:
    existing_metrics = pd.read_csv('models_ml/metrics/model_metrics.csv')
    updated_metrics = pd.concat([existing_metrics, metricsdf], ignore_index=True)
except FileNotFoundError:
    updated_metrics = metricsdf

updated_metrics.to_csv('models_ml/metrics/model_metrics.csv', index=False)
print("Métricas guardadas en 'model_metrics.csv'")

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC'], 
            y=[accuracy, precision, recall, f1, roc_auc])
plt.title('Métricas del Ensemble Model')
plt.ylim(0, 1)
plt.savefig('models_ml/graphics/em_metrics.png')
plt.close()
print("Gráfico de métricas guardado como 'em_metrics.png'")

# Guardar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'EM (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC EM')
plt.legend(loc="lower right")
plt.savefig('models_ml/graphics/roc_curve_em.png')
plt.close()
