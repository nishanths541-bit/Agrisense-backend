from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
from datetime import datetime

app = Flask(__name__)
import os

MODEL_PATH = os.path.join("models", "plant_disease_model.h5")
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___healthy",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy",
]

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")
prediction_history = []
CORS(app)
pesticide_map = {
    "Tomato___Early_blight": {"pesticide": "Mancozeb", "dosage": "2g/litre", "freq": "Every 7 days"},
    "Tomato___Late_blight": {"pesticide": "Metalaxyl+Mancozeb", "dosage": "2.5g/litre", "freq": "Every 5 days"},
    "Tomato___healthy": {"pesticide": "None needed", "dosage": "—", "freq": "—"},
    "Potato___Early_blight": {"pesticide": "Azoxystrobin", "dosage": "1ml/litre", "freq": "Every 10 days"},
    "Potato___Late_blight": {"pesticide": "Chlorothalonil", "dosage": "2ml/litre", "freq": "Every 7 days"},
    "Potato___healthy": {"pesticide": "None needed", "dosage": "—", "freq": "—"},
    "Corn_(maize)___Common_rust_": {"pesticide": "Propiconazole", "dosage": "1ml/litre", "freq": "Every 10 days"},
    "Corn_(maize)___healthy": {"pesticide": "None needed", "dosage": "—", "freq": "—"},
    "Strawberry___Leaf_scorch": {"pesticide": "Myclobutanil or Captan", "dosage": "1.5ml/litre", "freq": "Every 10 days"},
    "Strawberry___healthy": {"pesticide": "None needed", "dosage": "—", "freq": "—"},
    "Apple___Apple_scab": {"pesticide": "Captan", "dosage": "2g/litre", "freq": "Every 7 days"},
    "Apple___healthy": {"pesticide": "None needed", "dosage": "—", "freq": "—"},
    "Grape___Black_rot": {"pesticide": "Mancozeb", "dosage": "2g/litre", "freq": "Every 7 days"},
    "Grape___healthy": {"pesticide": "None needed", "dosage": "—", "freq": "—"},
}
prediction_history = []
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    # Receive GPS coordinates if provided
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    print("GPS Latitude:", latitude)
    print("GPS Longitude:", longitude)
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype=np.float32)
    )
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    class_name = CLASS_NAMES[class_idx]
    confidence = float(predictions[0][class_idx] * 100)
    


    suggestion = pesticide_map.get(class_name, {
        "pesticide": "Consult agricultural expert",
        "dosage": "—",
        "freq": "—"
    })
    prediction_history.append({
    "latitude": float(latitude) if latitude else None,
    "longitude": float(longitude) if longitude else None,
    "disease": class_name,
    "confidence": round(confidence, 2),
    "timestamp": datetime.now().isoformat()
})
    return jsonify({
        "disease": class_name,
        "confidence": round(confidence, 2),
        "pesticide": suggestion["pesticide"],
        "dosage": suggestion["dosage"],
        "frequency": suggestion["freq"]
    })


from flask import Flask, request, jsonify, render_template

@app.route('/prediction-history', methods=['GET'])
def get_prediction_history():
    return jsonify(prediction_history)

@app.route('/')
def home():
    return render_template("index.html")
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))