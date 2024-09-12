import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV
df = pd.read_csv('models_ml/metrics/model_metrics_ok.csv')

# Definir las métricas a graficar
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

# Crear el gráfico de barras
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df['Model']))
width = 0.15

# Graficar las barras para cada métrica
for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, df[metric], width, label=metric)
    
    # Agregar los valores encima de cada barra
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

# Ajustar las etiquetas y límites
ax.set_xticks(x + (len(metrics)-1)*width/2)
ax.set_xticklabels(df['Model'])
ax.set_xlabel('Modelo')
ax.set_ylabel('Métrica')
plt.ylim(0.8, 1.0)
plt.title('Comparación de Rendimiento de Modelos')

# Ajustar el espacio inferior para la leyenda
plt.subplots_adjust(bottom=0.2)

# Mover la leyenda debajo del gráfico con más margen
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(metrics))

# Guardar el gráfico
plt.savefig('models_ml/graphics/modelo_comparacion_ajustado.png')
plt.close()

