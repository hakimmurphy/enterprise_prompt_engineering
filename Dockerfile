
# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (including templates)
COPY . .

# Expose port for Flask
EXPOSE 5050

# Set environment variable for Flask
ENV FLASK_APP=app.py

# Default command to run Flask app
CMD ["python", "app.py"]
