# Importar bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Cargar el archivo CSV proporcionado
file_path = 'Airlines_Modified.csv'
df = pd.read_csv(file_path)

# Preprocesamiento de datos
df['Arrival Delay in Minutes'].fillna(0, inplace=True)
df['satisfaction'] = df['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

label_encoder = LabelEncoder()
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = df['satisfaction']

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation

# Modelo de Random Forest
rf_model = RandomForestClassifier(random_state=42)

# Evaluar el modelo utilizando validación cruzada
cv_scores_rf = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')

# Imprimir los resultados
print("Random Forest Cross-Validation Accuracy Scores:", cv_scores_rf)
print("Random Forest Mean Accuracy:", cv_scores_rf.mean())
print("Random Forest Standard Deviation:", cv_scores_rf.std())

# Modelo de Regresión Logística
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Evaluar el modelo utilizando validación cruzada
cv_scores_log_reg = cross_val_score(log_reg_model, X, y, cv=kf, scoring='accuracy')

# Imprimir los resultados
print("\nLogistic Regression Cross-Validation Accuracy Scores:", cv_scores_log_reg)
print("Logistic Regression Mean Accuracy:", cv_scores_log_reg.mean())
print("Logistic Regression Standard Deviation:", cv_scores_log_reg.std())
