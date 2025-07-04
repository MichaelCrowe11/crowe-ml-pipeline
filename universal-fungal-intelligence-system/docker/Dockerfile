FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies including RDKit dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    libfontconfig1 \
    libice6 \
    libxt6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies (install RDKit first)
RUN pip install --no-cache-dir rdkit-pypi
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set Python path
ENV PYTHONPATH=/app/src

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit web UI
CMD ["python", "run_web_ui.py", "--host", "0.0.0.0", "--port", "8501", "--docker"]