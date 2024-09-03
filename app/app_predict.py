from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo guardado
model = joblib.load('knn_model.pkl')
print("Modelo cargado correctamente.")

# Definir la ruta para predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos enviados en el cuerpo de la solicitud
    data = request.get_json(force=True)
    # Convertir los datos a un formato numpy array (asegúrate de que coincida con las características del modelo)
    input_features = np.array(data['features']).reshape(1, -1)

    # Realizar predicción
    prediction = model.predict(input_features)
    
    # Devolver la predicción como respuesta JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
