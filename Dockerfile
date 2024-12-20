FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for model and temporary uploads
RUN mkdir -p /app/model
RUN mkdir -p /app/temp_uploads

# Copy your model and application code
COPY MAIN_MUZZLE.h5 /app/model/
COPY app.py .

# Update the model path in the application
ENV MODEL_PATH=/app/model/MAIN_MUZZLE.h5

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
