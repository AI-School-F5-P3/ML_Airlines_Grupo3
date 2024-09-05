# Importar bibliotecas necesarias
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np


# Cargar el archivo CSV proporcionado
file_path = 'data/airline_clean_data.csv'
df = pd.read_csv(file_path)

# Separar características y etiquetas
X = df.drop(columns=['Arrival Delay in Minutes', 'satisfaction'])
y = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)  # Convertir a variable binaria si es necesario

# Configurar la validación cruzada con K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros para el Árbol de Decisión Clasificador
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2']
}

# Configurar la búsqueda de hiperparámetros para Árbol de Decisión Clasificador
dt_model = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=kf, scoring='accuracy', n_jobs=-1, verbose=2)

# Entrenar el modelo de Árbol de Decisión Clasificador con la búsqueda de hiperparámetros
grid_search_dt.fit(X, y)

# Obtener el mejor modelo encontrado
best_dt_model = grid_search_dt.best_estimator_

# Evaluar el modelo usando validación cruzada
cv_scores_acc = cross_val_score(best_dt_model, X, y, cv=kf, scoring='accuracy')

# Mostrar resultados promedio y desviación estándar
print(f'Best parameters found for Decision Tree: {grid_search_dt.best_params_}')
print(f'Cross-Validation Mean Accuracy: {np.mean(cv_scores_acc)}')
print(f'Cross-Validation Std Accuracy: {np.std(cv_scores_acc)}')

# Guardar resultados en un DataFrame para comparación futura
results = pd.DataFrame({
    'Model': ['Decision Tree'],
    'Best Parameters': [grid_search_dt.best_params_],
    'Test Accuracy': [np.mean(cv_scores_acc)]
})

# Guardar los resultados en un archivo CSV
results.to_csv('decision_tree_classification_results.csv', index=False)
print("Results saved to 'decision_tree_classification_results.csv'")
plt.figure(figsize=(200, 150))
plot_tree(best_dt_model, feature_names=X.columns, class_names=['neutral or dissatisfied', 'satisfied'], filled=True, max_depth=2)
plt.show()