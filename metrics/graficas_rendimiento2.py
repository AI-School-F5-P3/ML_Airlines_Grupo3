import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('metrics/model_metrics_ok.csv')

# Crear el gráfico de barras para AUC
plt.figure(figsize=(10, 6))
models = df['Model']
auc_values = df['ROC AUC']

bars = plt.bar(models, auc_values)
plt.ylim(0.9, 1.0)  # Ajustar el rango del eje y para mostrar mejor las diferencias
plt.title('Comparación de AUC ROC entre modelos')
plt.xlabel('Modelos')
plt.ylabel('AUC ROC')

# Añadir etiquetas de valor encima de cada barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('comparacion_auc_roc.png')
plt.show()