# Use a slim Python image to keep size down
FROM python:3.10-slim

# Install system dependencies required for OpenCV (used by Ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Environment variables should be passed at runtime, NOT hardcoded here
# But we can set a default host if needed
ENV PYTHONUNBUFFERED=1

# Command to run the bot
CMD ["python", "app_with_followup.py"]