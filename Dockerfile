# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with specific flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 0 app:app 