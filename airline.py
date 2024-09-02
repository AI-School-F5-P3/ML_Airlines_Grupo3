import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler #para variables numéricas
from sklearn.preprocessing import OneHotEncoder #para variables categóricas nominales (Gender	Customer Type	Type of Travel)
from sklearn.preprocessing import OrdinalEncoder #para variables categóricas ordinales (Class)
from sklearn.preprocessing import LabelEncoder #Codificación de etiquetas en variable objetivo
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
from IPython.display import display, HTML



#Carga de datos

df = pd.read_csv("airline_passenger_satisfaction.csv")
print("Dimensionalidad de los datos:", df.shape)
print(df.head())

#Descripción de datos

print(df.dtypes)
print(df.describe())


# Variables tipo "object"
categorical_variables = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

# Contar la cantidad en cada categoría
for variable in categorical_variables:
    category_counts = df[variable].value_counts()
    print(f"Categorías en {variable}:")
    print(category_counts)
    print()


#Limpieza y preparación de los datos 

df = df.drop(["Unnamed: 0", "id"], axis=1)

# Identificar valores nulos en el DataFrame
null_values = df.isnull()

# Sumar los valores nulos por columna
null_counts = null_values.sum()

# Visualizar las columnas con valores nulos y la cantidad de nulos por columna
print("Valores nulos por columna:")
print(null_counts)

# Identificar valores faltantes en el DataFrame
missing_values = df.isna()

# Sumar los valores faltantes por columna
missing_counts = missing_values.sum()

# Visualizar las columnas con valores faltantes y la cantidad de valores faltantes por columna
print("Valores faltantes por columna:")
print(missing_counts)

# Identificar valores duplicados en el DataFrame
duplicates = df[df.duplicated()]

# Mostrar las filas duplicadas
print("Filas duplicadas:")
print(duplicates)



#Escalamiento de variables numéricas:

# Identificar las columnas numéricas
numeric_features = ["Age", "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
                   "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
                   "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
                   "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
                   "Departure Delay in Minutes", "Arrival Delay in Minutes"]

# Crear un objeto RobustScaler
scaler = RobustScaler()

# Aplicar el escalado a las columnas numéricas
df[numeric_features] = scaler.fit_transform(df[numeric_features])


#Escalamiento de variables categóricas nominales:

# Identificar las columnas categóricas
categorical_features = ["Gender", "Customer Type", "Type of Travel"]

# Crear un objeto OneHotEncoder
encoder = OneHotEncoder(drop="first")

# Aplicar la codificación one-hot a las columnas categóricas
encoded_columns = pd.DataFrame(encoder.fit_transform(df[categorical_features]).toarray(), columns=encoder.get_feature_names_out(categorical_features))
df = pd.concat([df, encoded_columns], axis=1)
df.drop(categorical_features, axis=1, inplace=True)



#Escalamiento de variables categóricas ordinales:

# Identificar la columna categórica ordinal
ordinal_features = ["Class"]

# Definir el orden de las categorías
class_categories = [["Business", "Eco", "Eco Plus"]]

# Crear un objeto OrdinalEncoder con las categorías especificadas
ordinal_encoder = OrdinalEncoder(categories=class_categories)

# Aplicar el escalado a la columna categórica ordinal
df[ordinal_features] = ordinal_encoder.fit_transform(df[ordinal_features])


#Codificación de etiquetas en variable objetivo:

# Identificar la columna de la variable objetivo
target_feature = ["satisfaction"]

print(df[target_feature].head())
print(df[target_feature].shape)


# Crear un objeto LabelEncoder
label_encoder = LabelEncoder()

# Aplicar el escalado a la variable objetivo
df[target_feature] = label_encoder.fit_transform(df[target_feature])




#Manejo de valores nulos mediante imputación con algoritmo de aprendizaje automático:

# Paso 1: Separar los datos
# Datos de entrenamiento (sin valores nulos en "Arrival Delay in Minutes")
train_data = df.dropna(subset=["Arrival Delay in Minutes"])

# Datos a imputar (con valores nulos en "Arrival Delay in Minutes")
impute_data = df[df["Arrival Delay in Minutes"].isnull()]

# Paso 2: Seleccionar el modelo
model = LinearRegression()

# Paso 3: Entrenar el modelo
X = train_data.drop(["Arrival Delay in Minutes"], axis=1)
y = train_data["Arrival Delay in Minutes"]
model.fit(X, y)

# Paso 4: Predecir valores faltantes
X_impute = impute_data.drop(["Arrival Delay in Minutes"], axis=1)
predicted_values = model.predict(X_impute)

# Paso 5: Sustituir los valores nulos con las predicciones
df.loc[df["Arrival Delay in Minutes"].isnull(), "Arrival Delay in Minutes"] = predicted_values

# Verificar que no hay valores nulos restantes en "Arrival Delay in Minutes"
print("Valores nulos en Arrival Delay in Minutes:", df["Arrival Delay in Minutes"].isnull().sum())

print(df.shape)
"""
La verificación de la dimensionalidad es importante para garantizar que no se hayan perdido registros o 
columnas durante las etapas de preprocesamiento de datos."""


#2. Creación y evaluación del modelo
#2.1 Separación de variables

# Definir las variables predictoras (características)
X = df.drop("satisfaction", axis=1)

# Definir la variable objetivo
y = df["satisfaction"]


#2.2 División de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar las dimensiones de los conjuntos de entrenamiento y prueba
print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)
print("Dimensiones de y_train:", y_train.shape)
print("Dimensiones de y_test:", y_test.shape)

#2.3 Búsqueda de mejor modelo con hiperparámetros
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Crear un objeto de árbol de decisión
decision_tree = DecisionTreeClassifier(random_state=42)

# Definir los hiperparámetros que se desea ajustar y sus posibles valores
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Crear un objeto GridSearchCV para la búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='accuracy')

# Realizar la búsqueda de hiperparámetros en los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Obtener los mejores hiperparámetros
best_params = grid_search.best_params_

# Verificar los resultados
print("Mejor modelo:", best_model)
print("Mejores hiperparámetros:", best_params)

#2.4 Creación de objeto y entrenamiento del mejor modelo encontrado
# Crear un objeto de árbol de decisión con los mejores hiperparámetros encontrados
best_decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=20, min_samples_split=10, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
best_decision_tree.fit(X_train, y_train)

decision_tree.fit(X_train, y_train)

#2.5 Resultados en matriz de confusión
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Realizar predicciones en el conjunto de prueba
y_pred = decision_tree.predict(X_test)

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)

# Crear una figura
plt.figure(figsize=(8, 6))

# Usar Seaborn para visualizar la matriz de confusión
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)

# Configurar etiquetas y título
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusión')

# Mostrar la matriz de confusión
plt.show()

#2.6 Evaluación con métricas de rendimiento y reporte de clasificación 
from sklearn.metrics import accuracy_score, classification_report

# Calcular la exactitud
accuracy = accuracy_score(y_test, y_pred)
print("Exactitud:", accuracy)

# Generar el informe de clasificación que incluye precisión, recall y puntuación F1
report = classification_report(y_test, y_pred, target_names=['neutral or dissatisfied', 'satisfied'])
print("Informe de Clasificación:")
print(report)

# 3. Determinación de los atributos más importantes según el modelo
# 3.1 Gráfica del modelo

# Crear una figura de Seaborn
plt.figure(figsize=(200, 150))

# Graficar el árbol de decisión limitando la profundidad a 2 niveles
plot_tree(best_decision_tree, feature_names=X_train.columns, class_names=['neutral or dissatisfied', 'satisfied'], filled=True, max_depth=2)

# Mostrar el gráfico
plt.show()


#3.2	Obtención de los coeficientes de importancia por cada atributo, en base al criterio de pureza
# Obtener los coeficientes de importancia de cada atributo en base a la pureza
feature_importances = best_decision_tree.feature_importances_

# Crear un DataFrame para mostrar los coeficientes junto con los nombres de las características
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Ordenar el DataFrame por importancia en orden descendente
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Mostrar los coeficientes de importancia
print("Coeficientes de importancia:")
print(importance_df)

#4. Predicciones y probabilidad


# Calcular las probabilidades de predicción en el conjunto de prueba
probs = best_decision_tree.predict_proba(X_test)

# Crear un DataFrame para mostrar las probabilidades
probs_df = pd.DataFrame(probs, columns=['Probabilidad_neutral_or_dissatisfied', 'Probabilidad_satisfied'])

# Mostrar la tabla desplazable
display(HTML(probs_df.to_html()))