#!/bin/bash
set -e

echo "🔧 Updating apt and installing system dependencies..."
apt-get update && apt-get install -y \
    curl \
    wget \
    bzip2 \
    git \
    build-essential \
    ca-certificates \
    libgl1-mesa-glx \
    && apt-get clean

echo "📦 Installing Miniforge3..."
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /opt/miniforge3
rm /tmp/miniforge.sh

echo "⚙️ Configuring environment..."
# Add Miniforge to PATH for this shell
export PATH="/opt/miniforge3/bin:$PATH"
source /opt/miniforge3/etc/profile.d/conda.sh

# Add to bashrc (interactive shells)
echo 'export PATH="/opt/miniforge3/bin:$PATH"' >> /root/.bashrc
echo 'source /opt/miniforge3/etc/profile.d/conda.sh' >> /root/.bashrc

# Add to /etc/profile (login shells)
echo 'export PATH="/opt/miniforge3/bin:$PATH"' >> /etc/profile
echo 'source /opt/miniforge3/etc/profile.d/conda.sh' >> /etc/profile

echo "⚡ Installing Mamba..."
conda install mamba -n base -c conda-forge -y

# Detect actual workspace mount path (VSCode sets $WORKSPACE_FOLDER)
REQ_FILE="${WORKSPACE_FOLDER:-/workspaces/$(basename "$PWD")}/requirements.txt"

if [ -f "$REQ_FILE" ]; then
    echo "📜 Found requirements.txt at: $REQ_FILE"
    mamba install -n base -y -c conda-forge --file "$REQ_FILE"
else
    echo "❌ requirements.txt not found at expected path: $REQ_FILE"
fi


echo "✅ Setup complete! You can now run Python, Jupyter, etc."
