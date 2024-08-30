import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = 'airline_passenger_satisfaction.csv'  # Cambia esta ruta por la tuya
df = pd.read_csv(file_path)

# 1. Limpieza y revisión inicial
# Eliminar columnas innecesarias
df_cleaned = df.drop(columns=['Unnamed: 0', 'id'])


# Revisar las primeras filas
print(df_cleaned.head())

# Verificar el tipo de datos y la presencia de valores nulos
print(df_cleaned.info())
print(df_cleaned.isnull().sum())

# 2. Estadísticas descriptivas
desc_stats = df_cleaned.describe()
print(desc_stats)

# 3. Análisis de las variables categóricas
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    print(f'\nDistribución de {col}:')
    print(df_cleaned[col].value_counts())
    sns.countplot(x=col, data=df_cleaned)
    plt.title(f'Conteo de {col}')
    plt.show()

# 4. Análisis de las variables numéricas
numerical_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f'Distribución de {col}')
    plt.show()

# 5. Análisis de correlaciones
# Seleccionar solo las columnas numéricas para la matriz de correlación
numerical_columns_only = df_cleaned.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(12, 8))
correlation_matrix = df_cleaned[numerical_columns_only].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación')
plt.show()


# 6. Manejo de valores nulos
# Contar cuántos valores son cero en 'Departure Delay in Minutes'
num_zeros = (df_cleaned['Departure Delay in Minutes'] == 0).sum()
print(f"Número de ceros en 'Departure Delay in Minutes': {num_zeros}")
# Contar cuántos valores son cero en 'Arrival Delay in Minutes'
num_zeros2 = (df_cleaned['Arrival Delay in Minutes'] == 0).sum()
print(f"Número de ceros en 'Arrival Delay in Minutes': {num_zeros2}")


# Por ejemplo, para la columna 'Arrival Delay in Minutes', podrías llenar los valores nulos con la media:
df_cleaned['Arrival Delay in Minutes'].fillna(df_cleaned['Arrival Delay in Minutes'].mean(), inplace=True)

# Guardar el DataFrame limpio si es necesario
df_cleaned.to_csv('airline_passenger_satisfaction_cleaned.csv', index=False)