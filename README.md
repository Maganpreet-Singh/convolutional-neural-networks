# 🧠 Convolutional Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-8A2BE2?style=for-the-badge" alt="Deep Learning CNN">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

<p align="center">
  <strong>A hands-on learning repository for understanding, visualizing, and implementing Convolutional Neural Networks from first principles to practical computer vision.</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-learning-path">Learning Path</a> •
  <a href="#-repository-structure">Structure</a> •
  <a href="#-getting-started">Setup</a> •
  <a href="#-experiments">Experiments</a>
</p>

---

## 📌 Overview

This repository is dedicated to **Convolutional Neural Networks (CNNs)**, one of the foundational architectures in deep learning for visual data.

The objective is to move beyond simply calling `Conv2D()` and reporting an accuracy score. The focus is on understanding **what a convolution actually does**, how image tensors are transformed, why padding and stride matter, how feature maps are formed, and how these low-level operations become the building blocks of modern computer-vision systems.

The current repository contains a dedicated **Padding and Strides** module with a Jupyter notebook and supporting image assets. The notebook-driven structure is intended to grow into a complete CNN learning track over time. fileciteturn2file0

> **Learning philosophy:** `Theory → Mathematics → Code → Visualization → Experiment → Intuition`

A strong CNN practitioner should be able to explain both **what the code does** and **why the resulting tensor looks the way it does**.

---

## 🎯 What You Will Learn

By working through this repository, you will build intuition around:

- Image tensors and channel dimensions
- Kernels and learnable filters
- Convolution and feature extraction
- Padding and border handling
- Stride and spatial downsampling
- Feature-map dimensions
- Receptive fields
- Activation functions
- Pooling and feature compression
- CNN architecture design
- Forward propagation and backpropagation
- Loss functions and optimization
- Overfitting and regularization
- Data augmentation
- Model evaluation and error analysis
- Transfer learning and fine-tuning
- Practical image-classification workflows

---

# 🔥 Why CNNs?

An image is not just a long vector of unrelated numbers. Pixels have **spatial relationships**: neighboring pixels often describe parts of the same edge, texture, object, or structure.

A fully connected network can learn from images, but it does not naturally exploit locality and parameter sharing. CNNs introduce operations specifically suited to spatial data.

A simplified CNN pipeline looks like this:

```text
                    ┌──────────────────────┐
                    │      Input Image     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Convolution       │
                    │  Learn local patterns│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Activation        │
                    │       ReLU           │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Pooling / Stride   │
                    │  Reduce spatial size │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ More Conv Blocks     │
                    │ Learn richer features│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Flatten / Global Pool│
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Dense Classification │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │      Prediction      │
                    └──────────────────────┘
```

The exact architecture can vary dramatically, but the central principle is consistent: **learn useful spatial representations directly from data**.

---

# 🧩 Core Concepts

## 1. Image Representation

A grayscale image can be represented as a 2D tensor:

```text
Height × Width
```

An RGB image typically has three channels:

```text
Height × Width × 3
```

For a batch of RGB images:

```text
Batch × Height × Width × Channels
```

For example:

```text
32 × 224 × 224 × 3
```

means:

| Dimension | Meaning |
|---|---|
| `32` | Number of images in the batch |
| `224` | Height |
| `224` | Width |
| `3` | RGB channels |

Understanding shapes is not optional in deep learning. A surprising number of CNN bugs are really **tensor-shape bugs wearing a fake mustache**.

---

## 2. Convolution

A convolution layer applies a small learnable filter across local regions of an image.

Conceptually:

```text
Input
  ↓
Local Region
  ↓
Kernel / Filter
  ↓
Element-wise Multiplication
  ↓
Summation + Bias
  ↓
Feature Map Value
```

A simplified expression is:

```text
Feature Map = Input ⊛ Kernel + Bias
```

During training, the network learns kernel parameters using gradient-based optimization and backpropagation.

---

## 3. Kernels and Filters

A filter is designed to detect a particular pattern. In an actual trained CNN, the network **learns** the useful filter values rather than requiring the programmer to manually specify them.

Early learned filters may respond to patterns such as:

- Horizontal edges
- Vertical edges
- Diagonal structures
- Corners
- Simple textures

Deeper layers can combine lower-level features into increasingly complex visual representations.

A useful intuition is:

```text
Pixels
  ↓
Edges
  ↓
Textures
  ↓
Parts / Shapes
  ↓
Higher-level visual patterns
```

This is why CNN depth matters: later representations can be built from patterns discovered earlier.

---

# 🧱 Padding

Padding adds extra values around an input before applying the convolution.

Two common choices in TensorFlow/Keras are:

### `padding="valid"`

No padding is added.

```text
Input
  ↓
Convolution
  ↓
Smaller spatial dimensions
```

### `padding="same"`

Padding is chosen so that, for the common stride-1 case, the output spatial dimensions match the input spatial dimensions.

```text
Input
  ↓
Padding
  ↓
Convolution
  ↓
Same spatial size*
```

`*` The exact output dimensions depend on kernel size and stride.

### Why padding matters

Padding affects:

- Border information
- Output dimensions
- How many times edge pixels participate in computation
- How rapidly spatial dimensions shrink through a network

A model that repeatedly uses `valid` convolutions can reduce spatial dimensions much faster than one using suitable `same` padding.

---

# 🏃 Stride

**Stride** controls how far a convolution filter moves between positions.

```text
Stride = 1  → move one position at a time
Stride = 2  → move two positions at a time
```

Increasing the stride generally reduces spatial resolution and can also reduce computational cost in later layers.

Think of the stride as the **step size of the sliding window**.

---

# 📐 Convolution Output Formula

For one spatial dimension, a standard convolution-size relationship is:

```text
Output = floor((N + 2P - K) / S) + 1
```

where:

| Symbol | Meaning |
|---|---|
| `N` | Input size |
| `P` | Padding size |
| `K` | Kernel size |
| `S` | Stride |

### Example

Suppose:

```text
Input  = 28
Kernel = 3
Padding = 0
Stride = 1
```

Then:

```text
Output = floor((28 + 2(0) - 3) / 1) + 1
       = 26
```

So the spatial dimension changes from:

```text
28 → 26
```

This formula becomes extremely useful when designing architectures and debugging shape mismatches.

---

# 🗺️ Feature Maps

When a filter scans across an image, it produces a **feature map**.

A feature map can be interpreted as a spatial record of how strongly a particular learned pattern appears at different positions.

For example:

```text
                    ┌─ Filter 1 ─→ Feature Map 1
                    │
Input Image ─────────┼─ Filter 2 ─→ Feature Map 2
                    │
                    ├─ Filter 3 ─→ Feature Map 3
                    │
                    └─ Filter 4 ─→ Feature Map 4
```

Using multiple filters allows the network to learn different visual detectors simultaneously.

---

# 🧠 Receptive Field Intuition

A **receptive field** describes the portion of the original input that can influence a particular activation.

As layers are stacked, deeper neurons can indirectly depend on larger regions of the original image.

```text
Small local region
       ↓
Larger combined region
       ↓
Even larger contextual region
```

This is one reason why deeper CNNs can represent increasingly complex structures.

---

# 🧪 Current Repository Module

## `Padding and Strides`

The current repository includes:

```text
Padding and Strides/
├── CNN_Padding_and_Strides.ipynb
└── Images/
```

The notebook is the main learning resource currently present in the repository. fileciteturn2file0

### Notebook

**`CNN_Padding_and_Strides.ipynb`** focuses on developing practical intuition for the relationship between convolution configuration, padding, stride, and output representation.

Open it here:

[`Padding and Strides/CNN_Padding_and_Strides.ipynb`](./Padding%20and%20Strides/CNN_Padding_and_Strides.ipynb)

---

# 📁 Repository Structure

### Current structure

```text
convolutional-neural-networks/
│
├── Padding and Strides/
│   ├── CNN_Padding_and_Strides.ipynb
│   └── Images/
│
└── README.md
```

The repository currently contains the `Padding and Strides` directory, its Jupyter notebook, its image assets, and this README. fileciteturn1file0turn2file0

### Planned scalable structure

As the repository grows, a stronger organization can look like:

```text
convolutional-neural-networks/
│
├── 01_Foundations/
│   ├── image_tensors.ipynb
│   ├── convolution_basics.ipynb
│   └── feature_maps.ipynb
│
├── 02_Padding_and_Strides/
│   ├── CNN_Padding_and_Strides.ipynb
│   └── Images/
│
├── 03_CNN_Building_Blocks/
│   ├── activation_functions.ipynb
│   ├── pooling.ipynb
│   └── cnn_architecture.ipynb
│
├── 04_CNN_Training/
│   ├── forward_propagation.ipynb
│   ├── backpropagation.ipynb
│   ├── optimizers.ipynb
│   └── regularization.ipynb
│
├── 05_Computer_Vision/
│   ├── binary_classification/
│   └── multiclass_classification/
│
├── 06_Transfer_Learning/
│   ├── feature_extraction.ipynb
│   └── fine_tuning.ipynb
│
├── 07_Advanced_Vision/
│   ├── object_detection/
│   └── image_segmentation/
│
├── requirements.txt
├── LICENSE
└── README.md
```

This is a **future roadmap**, not a claim that those files already exist.

---

# 🗺️ Learning Path

The repository is best approached in layers rather than jumping directly into large architectures.

## Phase 1 — Foundations

Learn how image data is represented and manipulated.

```text
Images
  ↓
Tensors
  ↓
Channels
  ↓
Shapes
  ↓
Normalization
```

## Phase 2 — Convolution Fundamentals

Understand:

- Kernels
- Filters
- Convolution
- Padding
- Stride
- Output dimensions
- Feature maps

## Phase 3 — CNN Building Blocks

Move into:

- ReLU
- Max pooling
- Average pooling
- Flattening
- Global average pooling
- Dense layers

## Phase 4 — Training CNNs

Study:

- Forward propagation
- Loss functions
- Backpropagation
- Gradient descent
- Learning rates
- Batch size
- Epochs
- Validation
- Overfitting and underfitting

## Phase 5 — Improving Models

Add:

- Dropout
- Batch normalization
- Data augmentation
- Early stopping
- Learning-rate scheduling
- Better initialization

## Phase 6 — Evaluation

Go beyond accuracy:

| Metric | What it tells you |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | How reliable positive predictions are |
| Recall | How many actual positives are found |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Which classes are confused |
| Loss | Optimization objective |
| Validation metrics | How well the model generalizes |

For imbalanced datasets, class-wise metrics can be much more informative than accuracy alone.

## Phase 7 — Transfer Learning

Progress into pretrained CNNs and learn:

- Feature extraction
- Frozen layers
- Fine-tuning
- Domain adaptation
- Model comparison

Architectures worth studying include:

```text
VGG-style networks
ResNet
MobileNet
EfficientNet
```

## Phase 8 — Advanced Computer Vision

Eventually extend the repository toward:

- Object detection
- Image segmentation
- Model explainability
- Saliency and activation visualization
- Efficient inference
- Deployment

---

# 💻 Technology Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🧠 TensorFlow | Deep-learning framework |
| 🧩 Keras | High-level neural-network API |
| 📓 Jupyter Notebook | Interactive experimentation |
| 🔢 NumPy | Numerical computing |
| 📊 Matplotlib | Visualization |

The repository is designed around practical, notebook-based learning rather than theory alone.

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/Maganpreet-Singh/convolutional-neural-networks.git
cd convolutional-neural-networks
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install the core packages

```bash
pip install tensorflow numpy matplotlib jupyter
```

For reproducible projects, it is recommended to pin compatible package versions in a future `requirements.txt` file.

## 4. Launch Jupyter

```bash
jupyter notebook
```

Then open:

```text
Padding and Strides/CNN_Padding_and_Strides.ipynb
```

---

# 🔬 Experiments

The fastest way to build CNN intuition is to change one variable at a time and observe what happens.

## Experiment 1 — Padding

Compare:

```python
padding="valid"
padding="same"
```

Observe the output dimensions and border behavior.

## Experiment 2 — Stride

Compare:

```python
strides=1
strides=2
```

Observe how the spatial resolution changes.

## Experiment 3 — Kernel Size

Compare:

```text
3 × 3
5 × 5
7 × 7
```

Ask:

> How does the receptive field and output shape change?

## Experiment 4 — Number of Filters

Try:

```text
16 → 32 → 64 → 128
```

Observe how the number of output channels changes.

## Experiment 5 — Architecture Depth

Compare shallow and deeper models using:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Generalization gap

## Experiment 6 — Data Augmentation

Train with and without augmentation and compare validation behavior.

## Experiment 7 — Regularization

Compare models using:

```text
No regularization
      ↓
Dropout
      ↓
Batch Normalization
      ↓
Data Augmentation
```

The goal is not merely to get a higher number. The goal is to understand **why the model behaves differently**.

---

# 📊 CNN Model Evaluation Workflow

A practical computer-vision experiment should follow a disciplined pipeline:

```text
Dataset
   ↓
Train / Validation / Test Split
   ↓
Preprocessing
   ↓
Model Definition
   ↓
Training
   ↓
Validation
   ↓
Evaluation
   ↓
Error Analysis
   ↓
Iteration
```

For image classification, useful visual diagnostics include:

- Training vs validation loss
- Training vs validation accuracy
- Confusion matrix
- Misclassified examples
- Class distribution
- Prediction confidence
- Per-class metrics

A model that reaches strong training accuracy but poor validation accuracy is not automatically a successful model. That pattern may indicate **overfitting**.

---

# 🛡️ Reproducibility Principles

When adding future notebooks or projects, keep experiments easy to reproduce.

### Recommended practices

- Record major library versions.
- Use consistent dataset splits.
- Avoid data leakage.
- Set random seeds when deterministic comparisons are useful.
- Save important metrics and plots.
- Keep preprocessing consistent between training and inference.
- Document architectural choices.
- Clearly distinguish training, validation, and test results.

Reproducibility turns a notebook from a one-off experiment into a useful technical reference.

---

# 🧠 Important CNN Intuition

A useful mental model for CNN depth is:

```text
              INPUT IMAGE
                    │
                    ▼
          ┌───────────────────┐
          │ Shallow Conv Layer│
          └─────────┬─────────┘
                    │
             Edges / Lines
                    │
                    ▼
          ┌───────────────────┐
          │ Middle Conv Layers│
          └─────────┬─────────┘
                    │
           Textures / Patterns
                    │
                    ▼
          ┌───────────────────┐
          │  Deeper Layers    │
          └─────────┬─────────┘
                    │
        Complex visual structures
                    │
                    ▼
              CLASSIFIER
```

The exact learned representations depend on the dataset, model architecture, initialization, optimization, and training process. Still, the hierarchical representation idea is a powerful way to reason about CNNs.

---

# 🧮 Padding + Stride + Kernel: The Big Three

These three settings are worth memorizing conceptually:

| Component | Main question |
|---|---|
| Kernel | **What local region am I looking at?** |
| Padding | **What should happen at the borders?** |
| Stride | **How far should I move each step?** |

Then add filters:

| Component | Main question |
|---|---|
| Filters | **Which patterns should the network learn to detect?** |

Together they determine a large part of the spatial behavior of a convolution layer.

---

# 🧰 Recommended Future Additions

To evolve this repository into a strong CNN portfolio and learning resource, the highest-value next additions are:

1. A fundamentals notebook covering image tensors and convolution from scratch.
2. A dedicated pooling and activation-functions module.
3. Complete CNN architecture notebooks using TensorFlow/Keras.
4. A real image-classification project.
5. Training/validation visualization.
6. Confusion-matrix and misclassification analysis.
7. Data-augmentation experiments.
8. CNN comparison experiments.
9. Transfer-learning projects with pretrained models.
10. A clean inference/demo script.
11. `requirements.txt` for reproducible setup.
12. An explicit open-source `LICENSE`.

The long-term goal should be an **end-to-end computer-vision learning portfolio**, not just a collection of isolated notebooks.

---

# 🤝 Contributing

Contributions, suggestions, corrections, notebook improvements, and educational experiments are welcome.

A high-quality contribution should ideally contain:

- Clear explanations
- Reproducible code
- Meaningful visualizations
- Correct tensor dimensions
- A short explanation of observed results

For larger changes, describe the purpose and learning objective clearly so the repository remains coherent as it expands.

---

# 👨‍💻 Author

<p align="center">
  <strong>Maganpreet Singh</strong>
  <br>
  Deep Learning • Computer Vision • Machine Learning
  <br><br>
  <a href="https://github.com/Maganpreet-Singh">GitHub</a>
</p>

---

# ⭐ Support

If this repository helps you understand CNNs, consider giving it a ⭐ on GitHub.

A star is small, but it tells the maintainer the work is useful—and that is pretty solid fuel for the next notebook. 🚀

---

# 📜 License

No license file is currently present in the repository. Until an explicit license is added, reuse and redistribution rights should not be assumed.

For a public educational repository, adding an appropriate open-source license is recommended.

---

<p align="center">
  <strong>Learn the operation → visualize the tensor → inspect the feature map → understand the model → build the system. 🧠🔥</strong>
</p>
