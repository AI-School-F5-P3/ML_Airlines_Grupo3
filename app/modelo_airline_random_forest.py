import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.utils import shuffle

# Cargar el archivo CSV con el df limpio y escalado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Usar una muestra de 10.000 filas
sample_size = 10000
df_sample = shuffle(df, random_state=42).iloc[:sample_size]

# Dividimos el dataset en características (X) y variable objetivo (y) de la muestra (sample)
X = df_sample.drop(columns=['satisfaction'])
y = df_sample['satisfaction']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Definir el espacio de búsqueda de hiperparámetros para Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']  # Eliminando 'auto'
}

# Configurar la búsqueda de hiperparámetros para Random Forest
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# Entrenar el modelo de Random Forest con la búsqueda de hiperparámetros
grid_search_rf.fit(X_train, y_train)

# Evaluar el modelo ajustado en el conjunto de test
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Imprimir los resultados del modelo Random Forest
print("Best parameters found for Random Forest:", grid_search_rf.best_params_)
print("Random Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))  # Reporte de clasificación (incluye precision, recall y f1-score)
roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1])
print("Random Forest AUC:", roc_auc)

# Cálculo de métricas adicionales
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf, average='binary')  # Cambia 'binary' según el tipo de clasificación
recall = recall_score(y_test, y_pred_rf, average='binary')
f1 = f1_score(y_test, y_pred_rf, average='binary')

# Evaluar el modelo utilizando validación cruzada
cv_scores_rf = cross_val_score(best_rf_model, X, y, cv=kf, scoring='accuracy')

# Imprimir los resultados
print("Random Forest Cross-Validation Accuracy Scores:", cv_scores_rf)
print("Random Forest Mean Accuracy:", cv_scores_rf.mean())
print("Random Forest Standard Deviation:", cv_scores_rf.std())

# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Guardar el modelo en un archivo
joblib.dump(best_rf_model, 'models/rf_model.pkl')
print("Modelo guardado como rf_model.pkl")

# Métricas
metricsdf = pd.DataFrame({
    'Model': ['RF'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1_Score': [f1],
    'AUC_ROC': [roc_auc],
    'Best_Parameters': [str(grid_search_rf.best_params_)]
})

# Cargar métricas existentes (si las hay) y guardar en un archivo CSV
try:
    existing_metrics = pd.read_csv('metrics/model_metrics.csv')
    updated_metrics = pd.concat([existing_metrics, metricsdf], ignore_index=True)
except FileNotFoundError:
    updated_metrics = metricsdf

updated_metrics.to_csv('metrics/model_metrics.csv', index=False)
print("Métricas guardadas en 'model_metrics.csv'")

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC'], 
            y=[accuracy, precision, recall, f1, roc_auc])
plt.title('Métricas del Modelo Random Forest')
plt.ylim(0, 1)
plt.savefig('metrics/rf_metrics.png')
plt.close()
print("Gráfico de métricas guardado como 'rf_metrics.png'")
