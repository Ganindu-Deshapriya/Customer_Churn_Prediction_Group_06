FROM python:3.10-slim

WORKDIR /app

# Update package lists and install curl, gnupg (needed for Node.js setup)
RUN apt-get update && apt-get install -y curl gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y nodejs


# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    build-essential \
    curl \
    tzdata \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port for Jupyter
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
