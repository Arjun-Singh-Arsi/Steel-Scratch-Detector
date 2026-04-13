# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories
RUN mkdir -p uploads static/predictions

# Make port 7860 available to the world outside this container
# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the app with gunicorn
# Note: We bind to 0.0.0.0:7860 because HF expects the app there
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
