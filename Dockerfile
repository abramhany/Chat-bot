FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data in one layer
RUN python -m nltk.downloader punkt wordnet

# Copy the rest of the application
COPY . .

EXPOSE 8080
ENV PORT=8080

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 0 app:app