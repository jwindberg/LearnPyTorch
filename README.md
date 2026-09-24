# Learn PyTorch: Zero to Mastery 🚀

A hands-on, progressive curriculum for learning deep learning and neural network development using PyTorch. Designed for experimentation across different machines (Apple Silicon MPS, NVIDIA CUDA, or CPU).

---

## ⚡ Quickstart: Setup on Any Machine

Follow these steps to set up and run this repository on a new machine:

### 1. Clone the repository

```bash
git clone https://github.com/jwindberg/LearnPyTorch.git
cd LearnPyTorch
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note for NVIDIA GPU users (Linux / Windows):**
> If your system has an NVIDIA GPU and you need the CUDA-enabled build of PyTorch, install it directly following the [official PyTorch guide](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Verify your environment & hardware acceleration

Run the setup verification script:

```bash
python 00_verify_setup.py
```

This will confirm:
- Installed PyTorch version
- Available compute device:
  - **Apple Silicon:** `mps` (Metal Performance Shaders)
  - **NVIDIA GPU:** `cuda`
  - **Fallback:** `cpu`
- A quick tensor allocation test on your accelerated device.

---

## 📂 Repository Structure

```text
LearnPyTorch/
├── .gitignore              # Ignores virtual environments, cache, and OS artifacts
├── requirements.txt        # Core dependencies (torch, torchvision, numpy, matplotlib, etc.)
├── CURRICULUM.md           # Full syllabus and topic breakdown
├── README.md               # Setup and getting started guide
├── 00_verify_setup.py      # Verification script for environment & device acceleration
└── 01_tensor_basics.py     # Module 1: Tensor creation, shapes, arithmetic, device transfers
```

---

## 🗺️ Curriculum Overview

See [CURRICULUM.md](CURRICULUM.md) for the full 10-module roadmap:

| Module | Topic | Milestone |
| :--- | :--- | :--- |
| **0** | **Environment & Setup** | Verify environment & device acceleration (`00_verify_setup.py`) |
| **1** | **Tensors: The Foundation** | Shapes, broadcasting, indexing, operations (`01_tensor_basics.py`) |
| **2** | **Autograd & Optimization Mechanics** | Computational graphs, gradients, manual gradient descent |
| **3** | **Standard PyTorch Workflow (`torch.nn`)** | Modules, loss functions, optimizers, canonical training loop |
| **4** | **Non-Linearity & Deep MLPs** | Activation functions, multi-layer perceptrons, decision boundaries |
| **5** | **Data Engineering (`Dataset` & `DataLoader`)** | Batching, shuffling, custom dataset pipelines |
| **6** | **Computer Vision with CNNs** | Convolutions, pooling, image classification |
| **7** | **Transfer Learning** | Feature extraction & fine-tuning with pretrained models |
| **8** | **Sequential Models & Modern Architectures** | Embeddings, attention mechanisms, mini-Transformer block |
| **9** | **Model Lifecycle & Best Practices** | Checkpointing, inference mode, evaluation, deployment exports |

---

## 🔄 Daily Workflow / Syncing Across Machines

Before starting work:
```bash
git pull origin main
```

After completing exercises:
```bash
git add .
git commit -m "Complete module X exercises"
git push origin main
```
