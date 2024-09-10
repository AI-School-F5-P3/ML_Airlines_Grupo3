# Se utiliza imagen base de Python
FROM python:3.9-slim

# Se establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Se copian los archivos de la aplicación a la imagen de Docker
COPY . /app

# Se Instalan dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Se expone el puerto en el que corre Streamlit (por defecto es 8501)
EXPOSE 8501

# Se ejecuta la aplicación de Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
