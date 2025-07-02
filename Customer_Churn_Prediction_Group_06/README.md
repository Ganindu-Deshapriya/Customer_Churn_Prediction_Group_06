# 🧠 ML Practical Dev Environment (with Mamba + DevContainer)

This project provides a fully automated, cross-platform machine learning development environment using:

- 🐍 [Miniforge3](https://github.com/conda-forge/miniforge)
- ⚡️ [Mamba](https://github.com/mamba-org/mamba) (fast Conda)
- 🐳 Docker + VSCode DevContainers
- 🧪 JupyterLab-ready
- ✅ Works on **Windows**, **Linux**, and **macOS**

## 🏗️ Folder Structure

```
your-repo/
├── .devcontainer/
│   └── devcontainer.json
├── scripts/
│   └── setup-env.sh
├── requirements.txt
├── models/
│   └── sample_model.ipynb
├── README.md
```

## ⚙️ Getting Started

### ✅ 1. Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [VSCode](https://code.visualstudio.com/download)
- VSCode Extensions:
  - `Dev Containers` (`ms-vscode-remote.remote-containers`)
  - `Python` (`ms-python.python`)
  - `Jupyter` (`ms-toolsai.jupyter`)

### ▶️ 2. Open in Dev Container (First Time)

1. Clone this repo:
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
    cd YOUR_REPO
    code .
    ```

2. Open Command Palette in VSCode (`Ctrl+Shift+P`)

3. Select: `Dev Containers: Reopen in Container`

4. Wait while the container builds and runs `setup-env.sh`, which will:
   - Install Miniforge3
   - Initialize mamba shell integration
   - Install Python packages from `requirements.txt`

### 🛠️ 3. Update Dependencies (Manually)

If you **change or update `requirements.txt`**, you need to manually rerun the setup script **inside the container** terminal:

```bash
./scripts/setup-env.sh
```

> **Note**: If you get permission errors, run:
> ```bash
> chmod +x scripts/setup-env.sh
> ```

### 🧪 4. Launch Jupyter Lab

Inside the container terminal, run:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

Then open `http://localhost:8888` in your browser and paste the token shown in the terminal.

### 🔍 5. Testing Your Environment

To confirm your ML environment is correctly set up, run:

```bash
python -c 'import pandas; print("✅ pandas OK"); import sklearn; print("✅ scikit-learn OK"); import xgboost; print("✅ xgboost OK")'
```

You should see all three confirmation messages if the environment is ready.

### 🐛 6. Troubleshooting `requirements.txt` Not Found

If Python packages are not installed, it might be due to the script not finding your `requirements.txt`. How to check inside the container:

1. Run:
    ```bash
    pwd
    ls -l requirements.txt
    ```

2. Confirm `requirements.txt` exists in your current directory.

3. If it’s not found, either:
   - Change directory to the repo root where `requirements.txt` is located:
     ```bash
     cd /workspaces/your-repo-folder
     ```
   - Or specify the full path when manually installing:
     ```bash
     mamba install -n base --yes -c conda-forge --file /full/path/to/requirements.txt
     ```

## 📦 Python Dependencies

All Python package dependencies are listed in `requirements.txt`, which initially includes:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyterlab
- notebook
- ipykernel
- xgboost
- lightgbm
- catboost
- shap

Update this file as needed and rerun the setup script to apply changes.

## 📦 How the Setup Script Works

The `scripts/setup-env.sh` script:

- Installs system dependencies
- Downloads and installs Miniforge3 silently
- Runs `mamba shell init` to configure shell integration
- Adds necessary environment variables and source commands to `.bashrc` and `/etc/profile`
- Disables auto-activation of base environment by default
- Installs Python packages from `requirements.txt` using mamba

## 📋 Summary of Useful Commands

| Task | Command |
|------|---------|
| Reopen VSCode DevContainer | `Ctrl+Shift+P` → `Dev Containers: Reopen in Container` |
| Manually rerun setup script | `./scripts/setup-env.sh` (inside container terminal) |
| Activate conda base env (if needed) | `source ~/.bashrc && conda activate base` |
| Check Python packages | `python -c 'import pandas, sklearn, xgboost'` |
| Run Jupyter Lab | `jupyter lab --ip=0.0.0.0 --no-browser --allow-root` |

## ✅ Credits

This environment is based on an Ubuntu 22.04 container with Miniforge and mamba, to provide a fast, reproducible ML dev setup suitable for Windows, Mac, and Linux users.

If you want help adding sample notebooks or any further automation, just ask!

**Happy coding! 🚀**