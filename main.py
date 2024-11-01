from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np
import logging


app = FastAPI()
logging.basicConfig(level=logging.INFO) 

# Load the model
model = load_model("model.keras")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Disease Detection API!"}

# Endpoint for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")
    
    if not file.content_type.startswith('image/'):
        return {"error": "File type not supported. Please upload an image."}

    try:
        # Read image data from the uploaded file
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        
        # Preprocess the image to the input size required by your model
        image = image.resize((224, 224))  # Assuming 224x224 is the input size
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Get predictions from the model
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Define class labels based on your training
        classes = ["Healthy", "NotPlant", "Powdery", "Rust"]
        result = classes[predicted_class]

        logging.info(f"Prediction for {file.filename}: {result}")
        return {"prediction": result}

    except Exception as e:
        logging.error(f"Error processing {file.filename}: {str(e)}")
        return {"error": f"An error occurred while processing the image: {str(e)}"}
