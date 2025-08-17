# backend/app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy
import cv2
import uvicorn

from src.components.data_transformation import DataTransformation

# Initialize app
app = FastAPI(title="Traffic Sign Recognition API")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
try:
    model = tf.keras.models.load_model("artifacts/Saved_Models/VGGNet.h5")
except OSError:
    raise RuntimeError("Model file not found. Place 'VGGNet.h5' in backend folder.")

# List of class names for prediction results
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

def preprocess_image(image):
    gray_image = image.convert("L")  # Convert RGB to Grayscale
    normalized = np.array(gray_image) / 255.0  # Normalize to [0, 1]
    image_with_channel = np.expand_dims(normalized, axis=-1)  # Add channel dimension

    return image_with_channel

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load and preprocess image
        img = Image.open(file.file).convert("RGB").resize((32, 32))
        preprocessed_image = preprocess_image(img)
        img_array = np.expand_dims(preprocessed_image, 0)

        # Make prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))

        return {
            "class": class_names[class_idx],
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

