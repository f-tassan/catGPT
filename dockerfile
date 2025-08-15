FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    libgl1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p yolo_outputs /app/.config/Ultralytics

# Set permissions for YOLO config directory
RUN chmod 755 /app/.config/Ultralytics

# Expose the Gradio port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV YOLO_CONFIG_DIR=/app/.config/Ultralytics
ENV HOME=/app

# Run the application
CMD ["python", "app2.py"]