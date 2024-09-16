import pandas as pd
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import shuffle
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Cargar el archivo CSV con el df limpio y escalado
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
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Definir el espacio de búsqueda de hiperparámetros para Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None,'sqrt', 'log2']
}

# Configurar la búsqueda de hiperparámetros para Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Realizar la búsqueda de hiperparámetros con la muestra
print("Iniciando búsqueda de hiperparámetros...")
start_time = time.time()
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)


grid_search_rf.fit(X_sample, y_sample)
end_time = time.time()



print(f"Búsqueda de hiperparámetros completada en {end_time - start_time:.2f} segundos")
# Evaluar el modelo ajustado en el conjunto de test
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)




# Evaluar el modelo con validación cruzada
cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='accuracy')
y_pred = cross_val_predict(best_rf_model, X_test, y_test, cv=5)
y_pred_proba = cross_val_predict(best_rf_model, X_test, y_test, cv=5, method='predict_proba')[:, 1]

# Métricas con pos_label especificado
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)



# Evaluar el modelo en el conjunto de entrenamiento
y_pred_train = best_rf_model.predict(X_train)

# Métricas para el conjunto de entrenamiento
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, pos_label=1)
recall_train = recall_score(y_train, y_pred_train, pos_label=1)
f1_train = f1_score(y_train, y_pred_train, pos_label=1)

# Imprimir métricas para entrenamiento y prueba
print(f"Entrenamiento: Accuracy: {accuracy_train:.2f}, Precision: {precision_train:.2f}, Recall: {recall_train:.2f}, F1 Score: {f1_train:.2f}")
print(f"Prueba: Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")



# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])


# Verificación adicional sobre la precisión del modelo
accuracy = accuracy_score(y_test, y_pred_rf)
print(f"La exactitud del modelo Random Forest es de {accuracy * 100:.2f}%")




# Guardar el modelo en un archivo
joblib.dump(best_rf_model, 'models_ml/pkls/rf_model.pkl')
print("Modelo guardado como rf_model.pkl")



metricsdf = pd.DataFrame({
    'Model': ['Random Forest'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'ROC AUC': [roc_auc],
    'Best Parameters': [grid_search_rf.best_params_],
})

#Carga de df
try:
    existing_metrics = pd.read_csv('models_ml/metrics/model_metrics.csv')
    updated_metrics = pd.concat([existing_metrics, metricsdf], ignore_index=True)
except FileNotFoundError:
    updated_metrics = metricsdf

updated_metrics.to_csv('models_ml/metrics/model_metrics.csv', index=False)
print("Métricas guardadas en 'model_metrics.csv'")
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Random Forest')
plt.legend(loc="lower right")
plt.savefig('models_ml/graphics/roc_curve_rf.png')
plt.close()

