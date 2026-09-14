# 🧠 Convolutional Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-CNN-6C5CE7?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

<p align="center">
  <strong>A visual, experiment-driven deep-learning repository for understanding CNNs, implementing LeNet-5 on MNIST, and comparing convolutional networks with fully connected ANNs.</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-lenet-5">LeNet-5</a> •
  <a href="#-cnn-vs-ann">CNN vs ANN</a> •
  <a href="#-visual-analysis">Visual Analysis</a> •
  <a href="#-setup">Setup</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

---

## 📌 Overview

This repository is a hands-on study of **Convolutional Neural Networks (CNNs)** using Python, TensorFlow, Keras, Jupyter Notebook, NumPy, Pandas, Matplotlib, and scikit-learn.

The project is deliberately more than a collection of `Conv2D()` calls and a final accuracy number. It explores the mechanics of image representations, convolution, learned filters, feature maps, spatial transformations, model architecture, training behavior, classification metrics, confidence, feature spaces, parameter efficiency, and error analysis.

The current repository contains two major experiment tracks:

```text
convolutional-neural-networks/
│
├── CNN Vs ANN/
│   ├── ANN_Vs_CNN.ipynb
│   └── Images/
│
└── LeNet - 5/
    ├── LeNet_5.ipynb
    ├── lenet5_mnist.keras
    ├── lenet5_training_history.csv
    └── Images/
```

The repository tree confirms the dedicated CNN-vs-ANN notebook and the LeNet-5 notebook, saved model, training history, and extensive visualization assets. fileciteturn1file0 fileciteturn2file0

> **Learning philosophy:** Theory → Mathematics → Code → Visualization → Experiment → Evaluation → Interpretation.

---

# 🧭 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Core CNN intuition](#-core-cnn-intuition)
- [Image tensors](#-image-tensors)
- [Convolution](#-convolution)
- [Kernels and filters](#-kernels-and-filters)
- [Padding](#-padding)
- [Stride](#-stride)
- [Output-size mathematics](#-output-size-mathematics)
- [Feature maps](#-feature-maps)
- [Pooling](#-pooling)
- [Receptive fields](#-receptive-fields)
- [Hierarchical feature learning](#-hierarchical-feature-learning)
- [LeNet-5](#-lenet-5)
- [MNIST experiment](#-mnist-experiment)
- [CNN vs ANN](#-cnn-vs-ann)
- [Why CNNs fit images](#-why-cnns-fit-images)
- [Evaluation](#-evaluation)
- [Confidence analysis](#-confidence-analysis)
- [Feature-space analysis](#-feature-space-analysis)
- [Parameter efficiency](#-parameter-efficiency)
- [Visual analysis](#-visual-analysis)
- [Repository structure](#-repository-structure)
- [Setup](#-setup)
- [Notebook workflow](#-notebook-workflow)
- [Experiments to try](#-experiments-to-try)
- [Common mistakes](#-common-mistakes)
- [Reproducibility](#-reproducibility)
- [Future roadmap](#-future-roadmap)
- [Contribution](#-contribution)
- [License](#-license)

---

# 🎯 Objectives

The project is intended to develop practical intuition for:

- image tensors and channel dimensions;
- convolution and cross-correlation as implemented by common deep-learning libraries;
- kernel size, number of filters, padding, and stride;
- feature-map formation;
- pooling and spatial downsampling;
- receptive fields;
- activation functions;
- forward propagation;
- backpropagation and gradient-based optimization;
- CNN architecture design;
- model capacity and trainable parameters;
- training and validation curves;
- confusion matrices;
- per-class metrics;
- confidence distributions;
- learned filters and feature maps;
- feature-space visualization;
- PCA-based exploratory analysis;
- misclassification analysis;
- and CNN-versus-ANN trade-offs.

The repository's saved outputs show that the experiments go beyond final accuracy and include architecture, representation, training, confidence, and error-analysis visualizations. fileciteturn1file0 fileciteturn2file0

---

# 🧠 Core CNN intuition

An image contains spatial structure. Nearby pixels are usually related, and the same visual pattern can appear at many positions.

A CNN exploits that structure through **local connectivity** and **shared weights**.

A simplified pipeline is:

```text
Input image
     ↓
Convolution
     ↓
Activation
     ↓
Downsampling
     ↓
Convolution
     ↓
Activation
     ↓
Deeper representation
     ↓
Classifier
     ↓
Prediction
```

The key idea is representation learning:

```text
Raw pixels
    ↓
Simple visual patterns
    ↓
Combinations of patterns
    ↓
Higher-level representation
    ↓
Class prediction
```

A CNN therefore learns a transformation from raw image space into a representation that makes the task easier.

---

# 🖼️ Image tensors

A grayscale image can be represented as:

```text
Height × Width
```

A CNN input normally includes a channel dimension:

```text
Height × Width × Channels
```

For MNIST-style grayscale images:

```text
28 × 28 × 1
```

For RGB images:

```text
Height × Width × 3
```

A batch of RGB images commonly has the channels-last form:

```text
Batch × Height × Width × Channels
```

For example:

```text
32 × 224 × 224 × 3
```

means 32 images, each 224 pixels high, 224 pixels wide, with three color channels.

Shape reasoning is one of the most important practical CNN skills. Many apparent model errors are simply tensor-shape mismatches.

---

# 🔲 Convolution

A convolutional layer applies a small learned filter across local image regions.

Conceptually:

```text
Input patch
    ↓
Element-wise multiplication with kernel
    ↓
Summation
    ↓
Bias
    ↓
Activation value
```

A simplified notation is:

```text
Feature map = Input ⊛ Kernel + Bias
```

Strictly speaking, many deep-learning libraries implement cross-correlation rather than mathematical convolution because the kernel is not flipped, but the operation is conventionally called convolution in CNN terminology.

During training, the kernel values are learned by minimizing a loss function.

---

# 🧩 Kernels and filters

A kernel is a small spatial weight matrix. A `3 × 3` kernel, for example, examines a local nine-value neighborhood for a single-channel input.

A convolution layer contains multiple filters.

```text
                  ┌─ Filter 1 → Feature map 1
                  │
Input image ───────┼─ Filter 2 → Feature map 2
                  │
                  ├─ Filter 3 → Feature map 3
                  │
                  └─ Filter N → Feature map N
```

Early learned filters can respond to relatively simple structures such as edges, curves, and orientations. Deeper layers can combine earlier responses into richer representations.

The LeNet-5 experiment contains explicit visualizations of learned filters, including its early C1 filters. fileciteturn2file0

---

# 🧱 Padding

Padding adds values around the input boundary before applying a convolution.

## `valid`

```text
No explicit padding
→ spatial dimensions generally shrink
```

## `same`

```text
Padding selected to preserve spatial dimensions
for the common stride-1 case
```

Padding matters because it changes:

- output dimensions;
- how often border pixels participate;
- the rate at which spatial resolution shrinks;
- and the architecture's receptive-field behavior.

---

# 🏃 Stride

Stride determines how far the kernel moves between convolution positions.

```text
Stride = 1 → move one pixel at a time
Stride = 2 → move two pixels at a time
```

Increasing stride generally performs stronger spatial downsampling.

A useful mental model is:

> **The kernel says what pattern to look for; the stride says how far to move before looking again.**

---

# 📐 Output-size mathematics

For one spatial dimension, a standard convolution output formula is:

```text
Output = floor((N + 2P - K) / S) + 1
```

where:

| Symbol | Meaning |
|---|---|
| `N` | Input size |
| `P` | Padding |
| `K` | Kernel size |
| `S` | Stride |

Example:

```text
N = 28
K = 3
P = 0
S = 1
```

Therefore:

```text
Output = floor((28 + 0 - 3) / 1) + 1
       = 26
```

So:

```text
28 → 26
```

For two-dimensional data, calculate height and width separately.

This formula is worth memorizing because it makes architecture design much less mysterious.

---

# 🗺️ Feature maps

A feature map records where a learned filter responds strongly.

For example, a filter that responds to a certain edge orientation can produce high activations wherever that edge occurs.

This gives a spatial interpretation:

```text
Input
 ↓
Filter
 ↓
Activation map
```

Feature maps are therefore one of the most useful objects to visualize when learning CNNs.

The repository contains CNN and LeNet-5 feature-map visualizations, including CNN feature maps for a selected digit and LeNet-5 C3/S4 feature maps. fileciteturn1file0 fileciteturn2file0

---

# 🧺 Pooling

Pooling reduces spatial dimensions by summarizing local regions.

### Max pooling

Selects the maximum activation in a local window.

### Average pooling

Computes the average activation in a local window.

Pooling can reduce spatial resolution and computation while retaining useful information.

Modern CNNs also commonly use strided convolutions and global pooling, so classical pooling should be understood as one member of a broader family of spatial-reduction techniques.

---

# 🎯 Receptive fields

A neuron's receptive field is the region of the original input that can influence its activation.

Stacking layers increases the effective receptive field:

```text
Layer 1 → small local region
Layer 2 → larger effective region
Layer 3 → larger context
...
```

This helps explain why deeper CNN layers can represent patterns that require broader spatial context.

---

# 🧬 Hierarchical feature learning

A useful conceptual hierarchy is:

```text
Pixels
  ↓
Edges
  ↓
Corners / curves / textures
  ↓
Shapes and motifs
  ↓
Higher-level patterns
  ↓
Classification
```

This is a conceptual model, not a strict rule that every filter has one human-interpretable meaning. Neural representations can be distributed and complicated.

Still, the hierarchy is an excellent learning tool and becomes especially intuitive when paired with the repository's filter and feature-map visualizations.

---

# 🏛️ LeNet-5

LeNet-5 is a classic convolutional architecture associated with handwritten character recognition and the historical development of CNNs.

Its importance for this project is educational: it is compact enough to inspect layer by layer while still demonstrating the essential CNN pattern.

A simplified representation is:

```text
Input
  ↓
Convolution
  ↓
Subsampling
  ↓
Convolution
  ↓
Subsampling
  ↓
Fully connected representation
  ↓
Classification
```

The repository contains a dedicated LeNet-5 notebook, a saved Keras model, training history CSV, and a large set of LeNet-5 visualization artifacts. fileciteturn2file0

---

# 🔢 MNIST experiment

The LeNet-5 module uses the MNIST handwritten-digit problem as its central learning task.

A typical experiment pipeline is:

```text
Load MNIST
   ↓
Inspect samples
   ↓
Inspect class distribution
   ↓
Normalize pixels
   ↓
Add channel dimension
   ↓
Construct LeNet-5
   ↓
Train
   ↓
Record history
   ↓
Evaluate
   ↓
Generate predictions
   ↓
Analyze errors
   ↓
Inspect filters and feature maps
```

The directory includes visual outputs for MNIST samples, class distribution, pixel intensity analysis, model architecture, learned filters, feature maps, probabilities, training behavior, confusion matrices, confidence, and mistakes. fileciteturn2file0

---

# 📊 LeNet-5 analysis

The LeNet-5 image collection contains:

```text
32 X 32 LeNet input.png
LeNet-5 CNN Architecture.png
LeNet-5 C1 Learned Filters.png
3D Representation of LeNet-5 C1 Filters.png
LeNet-5 Early Feature Extraction.png
LeNet-5 C3 Feature Maps.png
LeNet-5 S4 Feature Maps.png
LeNet-5 Parameter Distribution.png
LeNet-5 Training Dashboard.png
LeNet-5 Training vs Validation Accuracy.png
LeNet-5 Training vs Validation Loss.png
LeNet-5 Confusion Matrix.png
Normalized Confusion Matrix.png
LeNet-5 Accuracy for Each Digit.png
LeNet-5 Correct-Class Confidence Distribution.png
LeNet-5 Correct-Class Confidence Distribution by Digit.png
Highest Confidence Predictions.png
Most Uncertain LeNet-5 Predictions.png
LeNet-5 Misclassified MNIST Images.png
```

These files are present in the repository. fileciteturn2file0

The important lesson is methodological: **train → inspect → diagnose → interpret**, rather than simply **train → report accuracy**.

---

# 📈 Training curves

Training curves show how metrics change across epochs.

Useful plots include:

```text
Training loss
Validation loss
Training accuracy
Validation accuracy
```

Typical interpretations:

### Both training and validation improve

The model is learning and generalizing reasonably well.

### Training improves while validation stalls or degrades

Potential overfitting.

### Both remain poor

Potential underfitting, inadequate optimization, insufficient capacity, poor preprocessing, or a problem in the data pipeline.

The repository includes LeNet-5 training-vs-validation accuracy and loss plots as well as CNN-vs-ANN learning curves. fileciteturn1file0 fileciteturn2file0

---

# ⚖️ CNN vs ANN

The second major project track is:

```text
CNN Vs ANN/
└── ANN_Vs_CNN.ipynb
```

The experiment compares an artificial neural network based on dense layers with a convolutional model on image data.

The repository includes corresponding analysis outputs for both architectures, including confusion matrices, learning curves, feature spaces, misclassified samples, parameter counts, training time, per-digit metrics, F1 score, confidence distributions, PCA, and performance dashboards. fileciteturn1file0

---

# 🧱 Why CNNs fit images

Consider a `28 × 28` grayscale image.

Flattening gives:

```text
28 × 28 = 784 values
```

An ANN can then process the image as a 784-dimensional vector.

A CNN instead keeps the spatial arrangement visible to convolutional layers.

```text
ANN
Image → Flatten → Dense → Dense → Output

CNN
Image → Conv → Activation → Downsample → Conv → Classifier
```

The CNN architecture therefore encodes an inductive bias suited to images.

---

# ♻️ Parameter sharing

A convolution filter uses the same learned weights at different spatial positions.

For example, one edge detector can search across an entire image.

This is fundamentally different from a dense layer where each input-output connection can have a distinct parameter.

Parameter sharing provides a powerful combination of:

```text
Local connectivity
+
Shared weights
+
Hierarchical feature extraction
```

The CNN-vs-ANN experiment explicitly includes trainable-parameter and accuracy-vs-parameter comparisons. fileciteturn1file0

---

# 🧮 Parameter counting

For a convolutional layer, the trainable parameter count is commonly:

```text
(kernel_height × kernel_width × input_channels × filters)
+ filters
```

when a bias exists for each filter.

Example:

```text
Kernel: 3 × 3
Input channels: 1
Filters: 32

Parameters = (3 × 3 × 1 × 32) + 32
           = 320
```

For comparison, a dense layer receiving 784 values and producing 32 outputs would require:

```text
784 × 32 + 32 = 25,120
```

This simple example illustrates why parameter sharing can dramatically change model size for image data.

---

# 📊 CNN vs ANN performance analysis

The repository includes visual outputs for:

- test accuracy;
- trainable parameters;
- training time;
- loss across epochs;
- F1 score by digit;
- per-digit accuracy;
- prediction confidence;
- confusion matrices;
- feature spaces;
- PCA explained variance;
- and a complete performance dashboard.

The corresponding assets are stored under `CNN Vs ANN/Images/`. fileciteturn1file0

This enables a multi-dimensional comparison:

```text
              Accuracy
                 │
                 │
Parameters ──────┼────── Training time
                 │
                 │
           Error profile
                 │
                 │
         Feature quality
```

The most accurate model is not automatically the best model for every deployment scenario.

---

# 🎯 Per-class evaluation

For ten-class digit recognition, overall accuracy can hide class-specific weaknesses.

A stronger evaluation asks:

```text
How well does the model classify 0?
How well does it classify 1?
...
How well does it classify 9?
```

The repository includes per-digit accuracy and F1-score visualizations in the CNN-vs-ANN analysis and a dedicated digit-accuracy visualization for LeNet-5. fileciteturn1file0 fileciteturn2file0

---

# 🧭 Confusion matrices

A confusion matrix records the relationship between true and predicted classes.

For a strong classifier, values are concentrated near the diagonal.

Off-diagonal values show which classes are confused.

The repository contains both ordinary and normalized confusion-matrix visualizations for the experiments. fileciteturn1file0 fileciteturn2file0

Normalization is useful when class frequencies differ or when you want to compare error proportions instead of raw counts.

---

# 🎲 Confidence analysis

A multiclass classifier can output probabilities such as:

```text
[0.01, 0.02, 0.90, 0.01, ...]
```

The maximum value is commonly used as the predicted class confidence.

But confidence should never be treated as proof of correctness.

A useful analysis asks:

- Are correct predictions generally high-confidence?
- Are incorrect predictions sometimes high-confidence?
- Which digits generate uncertainty?
- What do the most uncertain examples look like?

The LeNet-5 module includes confidence distributions, correct-class confidence plots, highest-confidence predictions, and most-uncertain predictions. fileciteturn2file0

The CNN-vs-ANN module also contains confidence-distribution and probability-feature-space visualizations. fileciteturn1file0

---

# ❌ Misclassification analysis

A wrong prediction is not merely a failed test example. It is a diagnostic opportunity.

A useful inspection table is:

```text
Image | True label | Predicted label | Confidence
```

Then ask:

```text
Is the sample visually ambiguous?
Does it resemble another class?
Is the model focusing on an irrelevant region?
Does the error occur repeatedly for the same class pair?
```

The repository includes dedicated misclassified-image visualizations for both CNN-vs-ANN and LeNet-5 analysis. fileciteturn1file0 fileciteturn2file0

---

# 🧬 Feature-space analysis

Deep networks transform inputs into learned representations.

Let:

```text
x → f(x)
```

where `f(x)` is the learned feature representation.

If representations from different classes become easier to separate, the final classifier has an easier job.

Because these representations can be high-dimensional, the project uses dimensionality reduction and 3D visualization to make them inspectable.

The CNN-vs-ANN directory contains static and interactive 3D feature-space artifacts and PCA explained-variance analysis. fileciteturn1file0

---

# 📉 PCA explained variance

Principal Component Analysis can project high-dimensional representations into a lower-dimensional coordinate system.

The explained-variance ratio tells us how much variance is retained by selected components.

The repository includes:

```text
ANN_vs_CNN_PCA_Explained_Variance.png
```

This is useful for exploratory representation analysis.

Important caveat:

> PCA maximizes retained variance, not class separability.

Therefore a PCA plot should support interpretation rather than be treated as a complete measure of representation quality.

---

# 🌐 Interactive feature spaces

The repository includes interactive HTML visualizations such as:

```text
ANN_Learned_Feature_Space_3D.html
ANN_Probability_Feature_Space_3D.html
ANN_vs_CNN_Accuracy_vs_Parameter_Count.html
ANN_vs_CNN_Learned_Feature_Space_3D.html
CNN_Learned_Feature_Space_3D.html
```

These files are part of the CNN-vs-ANN asset collection. fileciteturn1file0

Interactive plots can be especially useful for inspecting clusters, overlaps, outliers, and class boundaries that are difficult to see in a static screenshot.

---

# 🎨 Visual analysis

The project intentionally treats visualization as part of the experiment rather than decoration.

The saved assets cover several layers of analysis.

## Dataset level

```text
MNIST Sample Images.png
MNIST Training Class Distribution.png
MNIST_Class_Distribution.png
MNIST_Pixel_Intensity_Distribution.png
MNIST_Pixel_Intensity_Heatmap — Digit 5.png
```

## Architecture level

```text
LeNet-5 CNN Architecture.png
32 X 32 LeNet input.png
```

## Learned representation level

```text
LeNet-5 C1 Learned Filters.png
3D Representation of LeNet-5 C1 Filters.png
LeNet-5 Early Feature Extraction.png
LeNet-5 C3 Feature Maps.png
LeNet-5 S4 Feature Maps.png
CNN_First_Layer_Learned_Filters.png
CNN_Feature_Maps_Digit_7.png
```

## Training level

```text
LeNet-5 Training Dashboard.png
LeNet-5 Training vs Validation Accuracy.png
LeNet-5 Training vs Validation Loss.png
CNN_Learning_Curves.png
ANN_Learning_Curves.png
```

## Evaluation level

```text
Confusion matrices
Per-digit accuracy
F1 score
Confidence distributions
```

## Error-analysis level

```text
Misclassified images
Most uncertain predictions
Highest-confidence predictions
```

The actual repository tree contains these categories of assets. fileciteturn1file0 fileciteturn2file0

---

# 📁 Repository structure

```text
convolutional-neural-networks/
│
├── CNN Vs ANN/
│   ├── ANN_Vs_CNN.ipynb
│   └── Images/
│       ├── ANN visualizations
│       ├── CNN visualizations
│       ├── comparison plots
│       └── interactive HTML plots
│
├── LeNet - 5/
│   ├── LeNet_5.ipynb
│   ├── lenet5_mnist.keras
│   ├── lenet5_training_history.csv
│   └── Images/
│       ├── dataset analysis
│       ├── architecture diagrams
│       ├── learned filters
│       ├── feature maps
│       ├── training curves
│       ├── evaluation plots
│       ├── confidence analysis
│       └── error analysis
│
└── README.md
```

The exact repository tree confirms the two main modules and their saved artifacts. fileciteturn1file0 fileciteturn2file0

---

# 📦 Saved artifacts

## Trained model

```text
LeNet - 5/lenet5_mnist.keras
```

A saved Keras model is included in the repository. fileciteturn2file0

## Training history

```text
LeNet - 5/lenet5_training_history.csv
```

Training history is stored separately so metrics can be reloaded and analyzed without retraining. fileciteturn2file0

## Notebook

```text
LeNet - 5/LeNet_5.ipynb
```

The main LeNet-5 experiment is stored as a Jupyter notebook. fileciteturn2file0

## CNN-vs-ANN notebook

```text
CNN Vs ANN/ANN_Vs_CNN.ipynb
```

The main comparison experiment is stored separately. fileciteturn1file0

---

# 🛠️ Setup

## 1. Clone

```bash
git clone https://github.com/Maganpreet-Singh/convolutional-neural-networks.git
cd convolutional-neural-networks
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

A practical starting point is:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

For long-term reproducibility, pin the exact package versions used by the notebook environment in a `requirements.txt` file.

## 4. Start Jupyter

```bash
jupyter notebook
```

Then open:

```text
LeNet - 5/LeNet_5.ipynb
```

or:

```text
CNN Vs ANN/ANN_Vs_CNN.ipynb
```

---

# 🧪 Notebook workflow

A strong CNN notebook should follow a deliberate experimental sequence:

```text
1. Imports
2. Reproducibility setup
3. Dataset loading
4. Dataset inspection
5. Preprocessing
6. Visualization
7. Architecture definition
8. Model summary
9. Tensor-shape inspection
10. Training
11. Training curves
12. Test evaluation
13. Predictions
14. Confusion matrix
15. Class-wise metrics
16. Confidence analysis
17. Misclassification analysis
18. Learned filters
19. Feature maps
20. Feature-space visualization
21. Conclusions
```

The final stages are essential. Training a model tells you that the optimization procedure ran. Interpretation tells you something about what the model learned and where it struggles.

---

# 🔁 Forward propagation

During inference, data moves forward through the network:

```text
Input
 ↓
Convolution
 ↓
Activation
 ↓
Downsampling
 ↓
Deeper convolution
 ↓
Representation
 ↓
Classifier
 ↓
Output probabilities
```

Each stage transforms the representation.

The output is therefore the result of a chain of learned transformations rather than a direct lookup from pixels to labels.

---

# 🔄 Backpropagation

During training:

```text
Input
 ↓
Forward pass
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradients
 ↓
Parameter update
 ↓
Repeat
```

Backpropagation provides the gradients used by an optimizer to update convolution kernels, biases, dense-layer weights, and other trainable parameters.

This is the mechanism through which useful filters are learned from data.

---

# ⚡ Activation functions

Convolution and dense layers are linear transformations. Nonlinear activation functions allow stacked layers to represent nonlinear relationships.

A classic choice is ReLU:

```text
ReLU(x) = max(0, x)
```

Conceptually:

```text
negative input → 0
positive input → unchanged
```

Without nonlinearities, a stack of purely linear transformations would collapse into a single linear transformation.

---

# 🧮 Flattening

A tensor such as:

```text
7 × 7 × 64
```

contains:

```text
7 × 7 × 64 = 3136
```

values.

Flattening converts that tensor into a one-dimensional vector:

```text
7 × 7 × 64 → 3136
```

This vector can feed a dense classifier.

However, flattening can produce a large number of dense parameters when spatial feature maps are large. This is one reason modern architectures frequently use global pooling before classification.

---

# 🌍 Global average pooling

Global average pooling summarizes each channel across its spatial dimensions.

Conceptually:

```text
H × W × C
    ↓
1 × 1 × C
```

This can greatly reduce the number of parameters compared with flattening a large feature map.

---

# 🏋️ Training hyperparameters

Important CNN configuration choices include:

| Hyperparameter | Effect |
|---|---|
| Learning rate | Controls update magnitude |
| Batch size | Samples used per optimization step |
| Epochs | Number of training passes |
| Kernel size | Local spatial context |
| Filter count | Representation capacity |
| Stride | Spatial movement/downsampling |
| Padding | Border behavior/output shape |
| Pool size | Spatial reduction |
| Dropout | Regularization |
| Weight regularization | Controls model complexity |

The right value depends on the dataset, architecture, compute budget, and objective.

---

# 🛡️ Overfitting and generalization

A model can achieve excellent training performance while generalizing poorly.

Always compare:

```text
Training performance
vs
Validation/test performance
```

Common strategies include:

- data augmentation;
- dropout;
- weight regularization;
- batch normalization;
- early stopping;
- learning-rate schedules;
- reducing excessive model capacity;
- and improving the data distribution.

Do not automatically add every regularization method. Diagnose the failure first.

---

# 🧪 Experiments to try

The existing project is a strong foundation for controlled experiments.

## 1. Kernel-size experiment

Compare:

```text
3 × 3
5 × 5
```

Measure accuracy, parameter count, training time, and validation behavior.

## 2. Stride experiment

Compare:

```text
stride = 1
stride = 2
```

Track spatial dimensions and classification performance.

## 3. Padding experiment

Compare:

```text
valid
same
```

Record the output shape after every convolution.

## 4. Filter-count experiment

Compare:

```text
16
32
64
128
```

Plot capacity versus performance.

## 5. Depth experiment

Add convolution blocks gradually and observe representation quality.

## 6. Pooling experiment

Compare max pooling, average pooling, and strided convolutions.

## 7. Optimizer experiment

Compare optimizers while keeping the dataset, architecture, and evaluation protocol fixed.

## 8. Learning-rate experiment

Sweep the learning rate and plot convergence behavior.

## 9. Regularization experiment

Compare different regularization strategies using the same baseline architecture.

## 10. Confidence calibration

Compare predicted confidence against empirical correctness.

## 11. Representation evolution

Extract embeddings from several depths and compare class separation.

## 12. Robustness

Test controlled transformations such as translation, rotation, blur, noise, brightness, and contrast changes.

---

# 🔬 Robustness analysis

Clean test accuracy is useful but incomplete.

A stronger experiment can measure:

```text
Original accuracy
        ↓
Perturb image
        ↓
Re-run prediction
        ↓
Measure accuracy drop
        ↓
Measure confidence change
```

For example:

```text
Original
→ rotated
→ blurred
→ noisy
→ shifted
```

This begins to answer whether the learned representation is stable under realistic changes.

---

# 🔍 Explainability roadmap

The project can eventually be extended with:

- activation maximization;
- saliency maps;
- gradient-based attribution;
- class activation maps;
- Grad-CAM-style visualizations;
- layer-wise embedding inspection;
- and feature-response comparisons.

The goal is not to claim that a visualization is a perfect explanation. Instead, these techniques provide additional evidence about model behavior.

---

# 🚀 Transfer learning roadmap

After understanding LeNet-5 and CNN fundamentals, move toward pretrained models.

A typical workflow is:

```text
Pretrained backbone
       ↓
Freeze backbone
       ↓
Train task-specific head
       ↓
Evaluate
       ↓
Unfreeze selected layers
       ↓
Fine-tune
       ↓
Compare
```

Potential architectures to study include:

```text
VGG
ResNet
MobileNet
EfficientNet
```

This moves the project from classical CNN education toward modern applied computer vision.

---

# 🌐 Deployment roadmap

A complete machine-learning project can eventually progress from:

```text
Notebook
   ↓
Saved model
   ↓
Inference script
   ↓
API
   ↓
Web interface
   ↓
Containerized application
   ↓
Monitoring
```

Potential extensions include an image-upload classifier, Streamlit interface, FastAPI endpoint, or lightweight browser application.

The repository already contains a saved Keras model, making model-to-inference development a natural next step. fileciteturn2file0

---

# 🗂️ Recommended future repository structure

As the project grows, consider organizing future work like this:

```text
convolutional-neural-networks/
│
├── 01_Foundations/
│   ├── image_tensors.ipynb
│   ├── convolution_math.ipynb
│   └── feature_maps.ipynb
│
├── 02_Padding_Strides/
│   └── CNN_Padding_and_Strides.ipynb
│
├── 03_CNN_Building_Blocks/
│   ├── activations.ipynb
│   ├── pooling.ipynb
│   ├── normalization.ipynb
│   └── regularization.ipynb
│
├── 04_Classic_CNNs/
│   ├── LeNet_5/
│   ├── AlexNet/
│   └── VGG/
│
├── 05_CNN_Comparisons/
│   ├── ANN_vs_CNN/
│   ├── optimizer_comparison/
│   └── architecture_comparison/
│
├── 06_Transfer_Learning/
│   ├── ResNet/
│   ├── MobileNet/
│   └── EfficientNet/
│
├── 07_Applications/
│   ├── classification/
│   ├── detection/
│   └── segmentation/
│
├── assets/
├── requirements.txt
├── LICENSE
└── README.md
```

This is a proposed future structure, not a claim that all of these files currently exist.

---

# 🧠 Questions this project should help you answer

After working through the repository, you should be able to explain:

### Fundamentals

- What is a CNN?
- What is a kernel?
- What is a feature map?
- Why is locality useful?
- What is parameter sharing?

### Mathematics

- How is convolution output size calculated?
- How do padding and stride interact?
- How are convolution parameters counted?
- How does the receptive field grow?

### Architecture

- Why stack convolutional layers?
- Why downsample?
- Why use pooling?
- When might global pooling be preferable to flattening?

### Training

- What happens during forward propagation?
- How does backpropagation update filters?
- How does the learning rate affect convergence?
- How can overfitting be detected?

### Evaluation

- Why is accuracy not enough?
- What does a confusion matrix reveal?
- Why inspect per-class F1 scores?
- What does a confidence distribution tell you?

### Interpretation

- What do learned filters look like?
- How do feature maps change through depth?
- What does feature-space separation mean?
- Why can a model be confident and wrong?

### Engineering

- Which model uses fewer parameters?
- Which model trains faster?
- What is the accuracy/complexity trade-off?
- What should be measured before deployment?

---

# 🧪 Reproducibility

For repeatable experiments, control as many variables as practical.

A basic seed configuration is:

```python
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

For serious benchmarking, also record:

```text
Python version
TensorFlow version
CUDA / GPU environment
Dataset version
Model configuration
Batch size
Epoch count
Optimizer
Learning rate
Random seed
Hardware
Training duration
```

Exact reproducibility can still vary because of hardware and numerical implementation details, but explicit experiment records make results much easier to understand and reproduce.

---

# 🧹 Common mistakes

## Forgetting the channel dimension

A grayscale image may need:

```text
28 × 28 → 28 × 28 × 1
```

## Guessing tensor shapes

Calculate them or inspect the model summary.

## Comparing unequal models

Accuracy without parameter count, training time, and error analysis can produce a misleading comparison.

## Looking only at training accuracy

Training accuracy alone says little about generalization.

## Treating confidence as truth

A probability output is a model belief, not a guarantee.

## Ignoring misclassified images

Wrong predictions often reveal more about the model than correct predictions.

## Changing everything at once

If kernel size, optimizer, learning rate, architecture depth, and batch size all change together, you will not know what caused the result.

---

# 📈 A better experiment protocol

For meaningful comparisons:

```text
1. Define the question
2. Define the baseline
3. Change one major variable
4. Train under controlled conditions
5. Record metrics
6. Save plots
7. Inspect errors
8. Interpret results
9. Repeat
10. Document the conclusion
```

A good experiment does not merely produce a better score. It produces a better explanation.

---

# 🏆 Portfolio value

A strong CNN portfolio project should demonstrate more than API familiarity.

This repository can demonstrate:

```text
Python
  ↓
NumPy
  ↓
Tensor reasoning
  ↓
Machine-learning fundamentals
  ↓
TensorFlow / Keras
  ↓
CNN architecture
  ↓
Model training
  ↓
Visualization
  ↓
Evaluation
  ↓
Error analysis
  ↓
Representation learning
  ↓
Experiment design
```

The current project already contains the ingredients for this story: notebooks, a saved LeNet-5 model, training history, static visualizations, interactive feature-space plots, and a multi-metric CNN-vs-ANN comparison. fileciteturn1file0 fileciteturn2file0

---

# 🧭 Future roadmap

## Phase 1 — Foundations

- image tensors;
- convolution mathematics;
- padding;
- stride;
- feature maps.

## Phase 2 — CNN building blocks

- activation functions;
- pooling;
- normalization;
- dropout;
- regularization.

## Phase 3 — Classic architectures

- LeNet-5;
- AlexNet;
- VGG;
- Inception;
- ResNet.

## Phase 4 — Controlled experiments

- kernel-size sweeps;
- stride sweeps;
- parameter-budget comparisons;
- optimizer comparisons;
- learning-rate studies.

## Phase 5 — Interpretation

- learned filters;
- activation maps;
- feature embeddings;
- PCA;
- confidence calibration;
- error analysis.

## Phase 6 — Transfer learning

- ResNet;
- MobileNet;
- EfficientNet;
- feature extraction;
- fine-tuning.

## Phase 7 — Advanced computer vision

- object detection;
- segmentation;
- visual transformers;
- model compression;
- quantization;
- deployment.

---

# 🤝 Contribution

Contributions are welcome.

Good contributions improve one or more of:

```text
Correctness
Clarity
Visualization
Reproducibility
Experimental depth
Documentation
```

For notebook contributions:

1. Keep experiments focused.
2. Explain important tensor transformations.
3. Include meaningful visualizations.
4. Avoid unnecessary output clutter.
5. End with an interpretation or conclusion.

For structural changes:

1. Keep naming consistent.
2. Preserve the learning path.
3. Update documentation when files move.
4. Avoid committing unnecessary generated artifacts.

---

# 🐛 Suggested future issues

```text
[FEATURE] Add data augmentation study
[EXPERIMENT] Compare max and average pooling
[EXPERIMENT] Add optimizer benchmark
[EXPERIMENT] Add learning-rate sweep
[VISUALIZATION] Add activation maps
[INTERPRETABILITY] Add Grad-CAM-style analysis
[MODEL] Add ResNet transfer learning
[DEPLOYMENT] Add inference application
[DOCS] Add CNN mathematics notebook
[TESTING] Add automated shape checks
```

---

# 📄 License

No license was inferred or invented here. Add the intended license file and notice to the repository when you decide how the project should be licensed.

---

# 🙌 Final takeaway

A CNN is not simply a collection of convolutional layers.

It is a learned representation system.

The complete mental model is:

```text
Pixels
  ↓
Tensors
  ↓
Local neighborhoods
  ↓
Convolution kernels
  ↓
Feature maps
  ↓
Nonlinear transformations
  ↓
Spatial reduction
  ↓
Hierarchical representations
  ↓
Classifier
  ↓
Prediction
  ↓
Evaluation
  ↓
Error analysis
  ↓
Interpretation
  ↓
Improvement
```

The strongest lesson from this project is simple:

> **Architecture should match the structure of the data.**

Images have spatial structure. CNNs exploit that structure through local connectivity, shared weights, and hierarchical representation learning.

This repository makes those ideas tangible through LeNet-5, CNN-vs-ANN experiments, learned-filter visualizations, feature maps, 3D feature spaces, PCA analysis, training curves, parameter comparisons, confidence distributions, confusion matrices, per-class metrics, and misclassification analysis. fileciteturn1file0 fileciteturn2file0

If you can explain **why** each tensor changes shape, **what** each filter is doing, **how** the parameters are learned, **where** the model fails, and **why** one architecture is preferable under a given constraint, then you are no longer just using CNNs—you understand them.

---

<p align="center">
  <strong>🧠 Learn the operation. Visualize the representation. Measure the trade-off. Inspect the failure. Improve the model.</strong>
</p>
