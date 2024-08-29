import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Cargar los datos
data = pd.read_csv('airline_passenger_satisfaction.csv')

# Inspeccionar los primeros registros
print(data.head())

# Resumen descriptivo
print(data.describe(include='all'))


"""
Para calcular una matriz de correlación, es necesario que todas las columnas sean de tipo numérico.
"""


# Convertir las variables categóricas a dummy variables
data_encoded = pd.get_dummies(data, drop_first=True)  # drop_first=True evita el dummy variable trap



# Exportar el DataFrame modificado a un archivo Excel
data_encoded.to_excel('Airlines_Modified_2.xlsx', index=False)
print("Data successfully exported to 'Airlines_Modified_2.xlsx'.")


# Calcular la matriz de correlación
corr_matrix = data_encoded.corr()

# Visualizar la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()


# Visualización de la distribución de la satisfacción del cliente
plt.figure(figsize=(8, 6))
sns.countplot(x='satisfaccion', data=data_encoded)  # Reemplaza 'satisfaccion' con el nombre de tu columna objetivo
plt.title('Distribución de Satisfacción del Cliente')
plt.show()

# Visualización de variables categóricas
categorical_vars = ['categoria1', 'categoria2']  # Reemplaza con tus variables categóricas
for var in categorical_vars:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=var, hue='satisfaccion', data=data_encoded)  # Reemplaza 'satisfaccion' con tu variable objetivo
    plt.title(f'Distribución de {var} por Satisfacción')
    plt.show()

# Preparar los datos para el modelado
X = data_encoded.drop(columns=['satisfaccion'])  # Reemplaza 'satisfaccion' con el nombre de tu columna objetivo
y = data_encoded['satisfaccion']  # Reemplaza 'satisfaccion' con el nombre de tu columna objetivo

# Convertir variables categóricas a dummy variables
X = pd.get_dummies(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelado y evaluación con RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Importancia de las características
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Visualización de la importancia de las características
plt.figure(figsize=(12, 8))
plt.title('Importancia de las Características - Random Forest')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Modelado y evaluación con XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Importancia de las características con XGBoost
importances_xgb = xgb.feature_importances_
indices_xgb = np.argsort(importances_xgb)[::-1]

# Visualización de la importancia de las características con XGBoost
plt.figure(figsize=(12, 8))
plt.title('Importancia de las Características - XGBoost')
plt.bar(range(X_train.shape[1]), importances_xgb[indices_xgb], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices_xgb], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Evaluación del modelo
print('Evaluación del modelo RandomForest:')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print('Evaluación del modelo XGBoost:')
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
