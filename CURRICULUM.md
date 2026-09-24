# PyTorch & Machine Learning: Zero to Mastery

Welcome to your hands-on journey into Machine Learning (ML) and Deep Learning using **PyTorch**. This curriculum is structured as a progression from basic mathematical intuition and tensor operations to building, training, and deploying deep neural networks.

---

## 🗺️ Curriculum Overview

| Module | Title | Core Concepts | Hands-on Project / Milestone |
| :--- | :--- | :--- | :--- |
| **0** | **Environment & Setup** | Python virtual environments, PyTorch install, Apple Silicon (MPS) acceleration | Verify environment & GPU acceleration |
| **1** | **Tensors: The Foundation of PyTorch** | Tensor creation, shapes, dtypes, broadcasting, indexing, device transfers | Build a custom tensor arithmetic toolbox |
| **2** | **Autograd & Optimization Mechanics** | Computational graphs, gradients, `requires_grad`, `.backward()`, manual gradient descent | Solve linear regression with raw gradients |
| **3** | **The Standard PyTorch Workflow (`torch.nn`)** | `nn.Module`, layers, loss functions (`nn.MSELoss`, `nn.CrossEntropyLoss`), optimizers (`torch.optim`) | Train a linear regressor using PyTorch modules |
| **4** | **Non-Linearity & Deep Neural Networks (MLP)** | Activation functions (ReLU, GELU, Sigmoid), multi-layer perceptrons, decision boundaries | Non-linear binary & multi-class classification |
| **5** | **Data Engineering: Datasets & DataLoaders** | `torch.utils.data.Dataset`, `DataLoader`, batching, shuffling, data augmentations | Build a custom dataset pipeline for real data |
| **6** | **Computer Vision with CNNs** | Convolutions (`Conv2d`), pooling, receptive fields, batch normalization, dropout | Image classifier on FashionMNIST / CIFAR-10 |
| **7** | **Transfer Learning & Pretrained Models** | Feature extraction, fine-tuning, `torchvision.models` (e.g., ResNet, MobileNet) | Transfer learning on a custom image dataset |
| **8** | **Sequential Models & Modern Architectures** | Token embeddings, recurrent concepts, self-attention intuition, mini-Transformer block | Character-level language model / text generator |
| **9** | **Model Lifecycle, Checkpointing & Best Practices** | Saving/loading state dicts, inference mode, evaluation metrics, performance profiling | Export a trained model for deployment |

---

## 📚 Detailed Lesson Breakdown

### Module 0: Environment & Setup
- **Goal:** Set up an isolated Python environment and verify Apple Silicon (MPS) / hardware acceleration.
- **Topics:**
  - Setting up a virtual environment (`venv` or `conda`).
  - Installing `torch`, `torchvision`, `torchaudio`, and scientific libraries (`numpy`, `matplotlib`).
  - Checking hardware target: CPU vs Apple Metal (`torch.backends.mps.is_available()`).

### Module 1: PyTorch Tensors
- **Goal:** Master multidimensional arrays and tensor manipulation.
- **Topics:**
  - What is a Tensor? Scalars (0D), Vectors (1D), Matrices (2D), and N-D Tensors.
  - Creation functions: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`.
  - Data types (`torch.float32`, `torch.int64`, etc.) and conversions.
  - Reshaping: `.view()`, `.reshape()`, `.squeeze()`, `.unsqueeze()`, `.permute()`.
  - Broadcasting rules and matrix operations: dot products, element-wise arithmetic, `torch.matmul` / `@`.
  - Moving tensors between devices (`cpu` and `mps`).

### Module 2: Autograd — Automatic Differentiation
- **Goal:** Understand how PyTorch computes derivatives and drives learning.
- **Topics:**
  - The computational graph and dynamic graph execution.
  - Tracking gradients: `x.requires_grad = True`.
  - Computing derivatives: `loss.backward()` and inspecting `x.grad`.
  - Detaching tensors: `.detach()`, `with torch.no_grad():`.
  - Gradient accumulation and zeroing gradients (`x.grad.zero_()`).
  - *Exercise:* Implement gradient descent from scratch for $y = 3x + 2$.

### Module 3: PyTorch Neural Network API (`torch.nn`)
- **Goal:** Learn the canonical 5-step PyTorch training loop.
- **Topics:**
  - Subclassing `nn.Module` and implementing `__init__` and `forward()`.
  - Common layers: `nn.Linear`, `nn.Parameter`.
  - Loss functions: Mean Squared Error (`nn.MSELoss`) vs Cross-Entropy (`nn.CrossEntropyLoss`).
  - Optimizers: Stochastic Gradient Descent (`SGD`), Adam, AdamW.
  - The standard training loop:
    1. Forward pass: `y_pred = model(x)`
    2. Compute loss: `loss = loss_fn(y_pred, y_true)`
    3. Zero gradients: `optimizer.zero_grad()`
    4. Backward pass: `loss.backward()`
    5. Step optimizer: `optimizer.step()`

### Module 4: Non-Linearity & Deep Neural Networks (MLP)
- **Goal:** Break linear boundaries to solve complex classification problems.
- **Topics:**
  - Why linear layers alone cannot solve XOR or non-linear problems.
  - Activation functions: ReLU, LeakyReLU, GELU, Sigmoid, Softmax.
  - Constructing a Multi-Layer Perceptron (MLP) using `nn.Sequential` or modular classes.
  - Understanding logits vs probabilities.
  - Training loop with evaluation mode: `model.train()` vs `model.eval()`.

### Module 5: Data Pipelines (`Dataset` & `DataLoader`)
- **Goal:** Build scalable, batched data pipelines for training.
- **Topics:**
  - Subclassing `torch.utils.data.Dataset` (`__len__` and `__getitem__`).
  - Using `DataLoader` for batching, shuffling, and worker threads.
  - Splitting datasets: Train, Validation, and Test splits.
  - Normalization and data preprocessing.

### Module 6: Computer Vision with CNNs
- **Goal:** Build vision models capable of spatial feature hierarchy recognition.
- **Topics:**
  - Why MLPs fail on high-resolution images (parameter explosion, translation invariance).
  - Convolutions (`nn.Conv2d`): Kernel size, stride, padding.
  - Pooling layers: `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`.
  - Regularization: Dropout (`nn.Dropout`), Batch Normalization (`nn.BatchNorm2d`).
  - Measuring accuracy, precision, and building confusion matrices.

### Module 7: Transfer Learning
- **Goal:** Leverage state-of-the-art vision models trained on millions of images.
- **Topics:**
  - The intuition behind pretrained representations.
  - Loading pretrained architectures from `torchvision.models`.
  - Freezing backbone weights vs fine-tuning top classification heads.
  - Learning rate scheduling (`torch.optim.lr_scheduler`).

### Module 8: Sequential Data & Attention
- **Goal:** Introduction to processing sequential text/time-series data.
- **Topics:**
  - Tokenization and word embeddings (`nn.Embedding`).
  - Understanding sequence dimensions: `(batch_size, seq_len, embed_dim)`.
  - Intuition behind self-attention and Transformer blocks (`nn.MultiheadAttention`).
  - Building a miniature autoregressive character-level predictor.

### Module 9: Production & Best Practices
- **Goal:** Save, load, profile, and structure PyTorch code cleanly.
- **Topics:**
  - Saving and loading model state: `torch.save(model.state_dict(), path)` & `load_state_dict()`.
  - Reproducibility: Setting manual seeds (`torch.manual_seed`).
  - Writing modular project structures (config, dataset, model, train, eval).
  - Introduction to high-level ecosystems: PyTorch Lightning, Hugging Face Transformers.

---

## 🚀 How We Will Work Together
We will go step by step, mixing **concise theory**, **executable scripts**, and **interactive exercises** where you inspect the outputs and modify parameters.
