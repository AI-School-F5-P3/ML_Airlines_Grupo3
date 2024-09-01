# ML_Airlines_Grupo3

## Estructura del Proyecto

```plaintext
ML_Airlines_Grupo3/
│
├── app/                             # Directorio principal del backend
│   ├── __init__.py
│   ├── main.py                      # Archivo principal de FastAPI
│   ├── model.py                     # Código de carga y predicción del modelo ML
│   ├── schemas.py                   # Definición de los modelos de datos (pydantic models)
│   └── utils.py                     # Utilidades adicionales (opcional)
│
├── frontend/                        # Directorio principal del frontend
│   └── streamlit.py                 # Código de la aplicación Streamlit
│
├── models/                          # Directorio para los modelos entrenados
│   └── model.pkl                    # Archivo del modelo entrenado
│
├── data/                            # Directorio para datos (opcional)
│   ├── raw/                         # Datos originales
│   └── processed/                   # Datos procesados para el modelo
│
├── notebooks/                       # Directorio para Jupyter Notebooks
│   ├── exploratory_analysis.ipynb   # Notebook de análisis exploratorio
│   └── model_training.ipynb         # Notebook de entrenamiento del modelo
│
├── tests/                           # Directorio para pruebas unitarias y de integración
│   ├── test_main.py                 # Pruebas para la API de FastAPI
│   └── test_streamlit.py            # Pruebas para la aplicación Streamlit
│
├── Dockerfile                       # Archivo Docker para la aplicación
├── .dockerignore                    # Ignorar archivos innecesarios para Docker
├── .gitignore                       # Ignorar archivos innecesarios para Git
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Documentación del proyecto
```

## 
