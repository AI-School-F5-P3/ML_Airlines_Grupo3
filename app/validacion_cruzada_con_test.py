# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
import matplotlib.pyplot as plt

# Cargar el archivo CSV proporcionado
file_path = 'data/airline_passenger_satisfaction_model.csv'
df = pd.read_csv(file_path)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['satisfaction'])
y = df['satisfaction']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Evaluar el modelo utilizando validación cruzada en el conjunto de entrenamiento
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

# Imprimir los resultados de la validación cruzada
print("Random Forest Cross-Validation Accuracy Scores:", cv_scores_rf)
print("Random Forest Mean Accuracy:", cv_scores_rf.mean())
print("Random Forest Standard Deviation:", cv_scores_rf.std())

# Entrenar el modelo con todos los datos de entrenamiento
rf_model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = rf_model.predict(X_test)

# Calcular y mostrar las métricas para el modelo Random Forest
print("\nRandom Forest Test Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Calcular AUC y curva ROC
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print("Random Forest AUC: ", rf_auc)

# Graficar la curva ROC
fpr, tpr, _ = roc_curve(y_test, rf_probs)
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Modelo de Regresión Logística
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Evaluar el modelo utilizando validación cruzada en el conjunto de entrenamiento
cv_scores_log_reg = cross_val_score(log_reg_model, X_train, y_train, cv=kf, scoring='accuracy')

# Imprimir los resultados de la validación cruzada
print("\nLogistic Regression Cross-Validation Accuracy Scores:", cv_scores_log_reg)
print("Logistic Regression Mean Accuracy:", cv_scores_log_reg.mean())
print("Logistic Regression Standard Deviation:", cv_scores_log_reg.std())

# Entrenar el modelo con todos los datos de entrenamiento
log_reg_model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_log_reg = log_reg_model.predict(X_test)

# Calcular y mostrar las métricas para el modelo de Regresión Logística
print("\nLogistic Regression Test Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Calcular AUC y curva ROC
log_reg_probs = log_reg_model.predict_proba(X_test)[:, 1]
log_reg_auc = roc_auc_score(y_test, log_reg_probs)
print("Logistic Regression AUC: ", log_reg_auc)

# Graficar la curva ROC
fpr, tpr, _ = roc_curve(y_test, log_reg_probs)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {log_reg_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()
