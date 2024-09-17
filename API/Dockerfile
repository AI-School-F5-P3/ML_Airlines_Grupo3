# Utilizar una imagen base con Python
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la API al contenedor
COPY ./API /app/API

# Copiar el archivo requirements.txt al directorio de trabajo
COPY ./requirements.txt .

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto que usa FastAPI (8000)
EXPOSE 8000

# Comando para ejecutar FastAPI (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
