# Use an official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all app files
COPY . .

# Expose Streamlit port
EXPOSE 8080

# Streamlit command to run app.py
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]

