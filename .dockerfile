FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and model
COPY requirements.txt .
COPY model.onnx .
COPY main.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]