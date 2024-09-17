# Utilizar una imagen base con Python
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del frontend al contenedor
COPY . /app/frontend

# Copiar el archivo requirements.txt al directorio de trabajo
COPY ./requirements.txt .

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa Streamlit (8501)
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
