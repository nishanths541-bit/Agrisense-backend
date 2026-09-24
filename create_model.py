"""
create_model.py
----------------
Idha ஒரே ஒரு முறை run pannunga (unga laptop-la, Agrisense folder-lendhu).
Idhu oru MobileNetV2-base model create panni, "models/plant_disease_model.h5"
nu save pannum.

NOTE: Idhu TRAIN pannala — so predictions accurate-a irukkaathu (almost random).
Aana app.py "model loaded" nu kaattum, so full pipeline (upload -> predict ->
result -> map) work aagum for demo/testing purposes.

Run pannradhukku munnadi terminal-la:
    pip install tensorflow

Appuram:
    python create_model.py
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Same order as CLASS_NAMES in app.py — mாத்தினா inga um match pannunga
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

print("Building MobileNetV2-based model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"  # pretrained on general images, NOT plant diseases
)
base_model.trainable = False  # freeze base layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

os.makedirs("models", exist_ok=True)
save_path = os.path.join("models", "plant_disease_model.h5")
model.save(save_path)

print(f"\nDone! Model saved to: {save_path}")
print("WARNING: This model is NOT trained on your disease dataset.")
print("Predictions will not be accurate until you train it on real leaf images.")