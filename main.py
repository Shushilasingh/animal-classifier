from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load trained model
model = load_model("animal_classifier_model.keras")
class_names = ["cat", "dog", "elephant", "lion", "tiger"]

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    input_tensor = preprocess(image)
    prediction = model.predict(input_tensor)
    label = class_names[np.argmax(prediction)]
    return {"prediction": label}





