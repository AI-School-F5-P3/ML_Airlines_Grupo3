import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV
df = pd.read_csv('metrics/model_metrics.csv')

# Definir las métricas a graficar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df['Model']))
width = 0.15

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, df[metric], width, label=metric)

ax.set_xticks(x + (len(metrics)-1)*width/2)
ax.set_xticklabels(df['Model'])
ax.set_xlabel('Modelo')
ax.set_ylabel('Métrica')
ax.legend()
plt.title('Comparación de Rendimiento de Modelos')
plt.savefig('modelo_comparacion.png')
plt.close()