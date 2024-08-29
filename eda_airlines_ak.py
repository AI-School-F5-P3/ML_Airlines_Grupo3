import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

"""EDA"""

# Cargar los datos
df = pd.read_csv('airline_passenger_satisfaction.csv')

# Mostrar las primeras filas del dataset
print(df.head())

# Resumen estadístico de las características
print(df.describe())

# Comprobar valores nulos
print(df.isnull().sum())


#Reemplazar los 310 valores nulos de la columna 'Arrival Delay in Minutes'

# 1. Calcular la media de la columna 'Arrival Delay in Minutes'
mean_arrival_delay = df['Arrival Delay in Minutes'].mean()

# 2. Reemplazar los valores nulos en la columna 'Arrival Delay in Minutes' con la media calculada
df['Arrival Delay in Minutes'].fillna(mean_arrival_delay, inplace=True)

# Comprobar valores nulos luego del reemplazo de valores de la columna 'Arrival Delay in Minutes'
print(df.isnull().sum())


""" A continuación las columnas que tienen datos (valores) categóricos de tipo string como Gender, Customer Type, 
Type of Travel, Class y satisfaction, serán reemplazadas por datos del tipo integer (números).
 """
#Opción A

# # Reemplazar valores categóricos en la columna 'Gender'
# df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# # Reemplazar valores categóricos en la columna 'Customer Type'
# df['Customer Type'] = df['Customer Type'].map({'Loyal Customer': 0, 'Disloyal Customer': 1})

# # Reemplazar valores categóricos en la columna 'Type of Travel'
# df['Type of Travel'] = df['Type of Travel'].map({'Business travel': 0, 'Personal Travel': 1})

# # Reemplazar valores categóricos en la columna 'Class'
# df['Class'] = df['Class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})

# # Reemplazar valores categóricos en la columna 'satisfaction'
# df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})


#Opcion B


# Crear una instancia de LabelEncoder
labelencoder = LabelEncoder()

# Aplicar LabelEncoder a cada columna categórica
df['Gender'] = labelencoder.fit_transform(df['Gender'])
df['Customer Type'] = labelencoder.fit_transform(df['Customer Type'])
df['Type of Travel'] = labelencoder.fit_transform(df['Type of Travel'])
df['Class'] = labelencoder.fit_transform(df['Class'])
df['satisfaction'] = labelencoder.fit_transform(df['satisfaction'])


# Mostrar las primeras filas del DataFrame para verificar el cambio
print(df.head())

# Exportar el DataFrame modificado a un archivo Excel
df.to_excel('Airlines_Modified.xlsx', index=False)
print("Data successfully exported to 'Airlines_Modified.xlsx'.")


# Visualización de la distribución de la variable objetivo
sns.countplot(x='satisfaction', data=df)
plt.show()


#Exportar info a Excel

# Calcular estadísticas descriptivas
describe_df = df.describe()

# Crear un archivo Excel con pandas ExcelWriter
with pd.ExcelWriter('airlines_data_analysis.xlsx', engine='xlsxwriter') as writer:
    # Exportar las estadísticas descriptivas a una hoja de Excel
    describe_df.to_excel(writer, sheet_name='Describe')

    # Crear una figura de matplotlib para la variable objetivo (ejemplo: "satisfaction")
    plt.figure(figsize=(10, 6))
    
    # Graficar la distribución de la variable objetivo (ajusta "satisfaction" según corresponda)
    df['satisfaction'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Satisfaction')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Frequency')

    # Guardar el gráfico como una imagen
    plt.savefig('satisfaction_distribution.png')

    # Insertar la imagen del gráfico en la hoja de Excel
    worksheet = writer.sheets['Describe']
    worksheet.insert_image('H2', 'satisfaction_distribution.png')

print("Data and plot successfully exported to 'airlines_data_analysis.xlsx'.")

