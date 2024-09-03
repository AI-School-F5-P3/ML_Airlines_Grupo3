# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Cargar el archivo CSV proporcionado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Separar características y etiquetas
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros para Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
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
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest AUC:", roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1]))

# Definir el espacio de búsqueda de hiperparámetros para Regresión Logística
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 200, 300]
}

# Configurar la búsqueda de hiperparámetros para Regresión Logística
log_reg_model = LogisticRegression(random_state=42)
grid_search_lr = GridSearchCV(estimator=log_reg_model, param_grid=param_grid_lr, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# Entrenar el modelo de Regresión Logística con la búsqueda de hiperparámetros
grid_search_lr.fit(X_train, y_train)

# Evaluar el modelo ajustado en el conjunto de test
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(X_test)

# Imprimir los resultados del modelo de Regresión Logística
print("Best parameters found for Logistic Regression:", grid_search_lr.best_params_)
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Logistic Regression AUC:", roc_auc_score(y_test, best_lr_model.predict_proba(X_test)[:, 1]))

# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, best_lr_model.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1]))
plt.plot(fpr_lr, tpr_lr, color='green', lw=2, label='Logistic Regression (AUC = %0.2f)' % roc_auc_score(y_test, best_lr_model.predict_proba(X_test)[:, 1]))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
