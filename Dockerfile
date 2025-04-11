FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with specific flags for blis
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir blis==0.7.9 --no-binary blis

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"] 