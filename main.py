import io
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime
import numpy as np
from PIL import Image
import uvicorn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dr-api")
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="API for detecting diabetic retinopathy from retinal images",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #  frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
try:
    logger.info("Loading ONNX model...")
    session = onnxruntime.InferenceSession('model.onnx')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    session = None  
@app.get("/health")
async def health_check():
    if session is None:
        return {"status": "unhealthy", "message": "Model failed to load"}
    return {"status": "healthy", "model_loaded": True}
def transform_image(image):
    """Preprocess image for model inference"""
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.5353, 0.3628, 0.2486], dtype=np.float32)
    std = np.array([0.2126, 0.1586, 0.1401], dtype=np.float32)
    img_array = (img_array - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0).astype(np.float32)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict diabetic retinopathy from retinal image
    
    - **file**: Upload a retinal image file
    
    Returns detailed classification for all DR grades and a binary classification
    """
    logger.info(f"Received image: {file.filename}, content-type: {file.content_type}")
    if session is None:
        raise HTTPException(status_code=503, detail="Model not available")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        image_data = await file.read()
        input_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform_image(input_img)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        logger.info("Running inference")
        prediction = session.run([output_name], {input_name: input_tensor})[0][0]
        exp_preds = np.exp(prediction - np.max(prediction))
        probabilities = exp_preds / exp_preds.sum()
        
        # Format results
        full_confidences = {labels[i]:float(f"{probabilities[i] * 100:.0f}") for i in labels}
        #full_confidences = {labels[i]: int(probabilities[i] * 100) for i in labels}
        #full_confidences = {labels[i]: f"{round(probabilities[i] * 100, 0)}" for i in labels}
        #full_confidences = {labels[i]: float(probabilities[i]) for i in labels}
        
        # Calculate binary classification
        #severe_prob = (full_confidences["Severe"] + 
             #         full_confidences["Moderate"] + 
                #      full_confidences["Proliferative DR"])
        
       # binary_result = {
          #  "No DR": full_confidences["No DR"],
           # "DR Detected": severe_prob
       # }
        
        highest_class = max(full_confidences.items(), key=lambda x: x[1])[0]
        logger.info(f"Prediction complete: highest probability class = {highest_class}")
        
        # Return both full and binary classifications
        return {
            "detailed_classification": full_confidences,
          #  "binary_classification": binary_result,
            "highest_probability_class": highest_class
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)