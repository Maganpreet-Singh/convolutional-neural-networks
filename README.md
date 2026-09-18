# 🧠 Convolutional Neural Networks — From Pixels to Learned Representations\n\n<p align="center"><strong>A visual, experiment-driven deep-learning repository covering CNN mechanics, classical architectures, representation learning, model comparison, and practical image classification.</strong></p>

<p align="center">🧠 Theory → 🧮 Mathematics → 💻 Implementation → 📊 Evaluation → 🔬 Interpretation → 🚀 Application</p>\n\n---\n\n## 🌌 Overview\n\nThis repository is a hands-on study of Convolutional Neural Networks using Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, scikit-learn, and Jupyter notebooks. The current repository contains dedicated modules for Padding and Strides, Pooling Layers, LeNet-5 on MNIST, CNN vs ANN analysis, and Cats vs Dogs classification.

The project deliberately goes beyond a single accuracy score. It includes training histories, saved Keras models, learned-filter visualizations, feature maps, confusion matrices, confidence analysis, per-class metrics, parameter comparisons, training-time comparisons, feature-space plots, PCA analysis, and misclassification inspection.

The core philosophy is simple: **a neural network should be studied as a representation-learning system, not treated as a black-box API call.**\n\n---\n\n## 📚 Table of Contents\n\n- [Repository Structure](#-repository-structure)
- [Learning Objectives](#-learning-objectives)
- [Why CNNs](#-why-cnns)
- [Image Tensors](#-image-tensors)
- [Convolution](#-convolution)
- [Kernels and Filters](#-kernels-and-filters)
- [Feature Maps](#-feature-maps)
- [Padding](#-padding)
- [Stride](#-stride)
- [Output Size](#-output-size)
- [Pooling](#-pooling)
- [Receptive Fields](#-receptive-fields)
- [Hierarchical Features](#-hierarchical-features)
- [Activation Functions](#-activation-functions)
- [LeNet-5](#-lenet-5)
- [MNIST](#-mnist)
- [CNN vs ANN](#-cnn-vs-ann)
- [Cats vs Dogs](#-cats-vs-dogs)
- [Evaluation](#-evaluation)
- [Confidence](#-confidence)
- [Feature Space](#-feature-space)
- [Error Analysis](#-error-analysis)
- [Training History](#-training-history)
- [Setup](#-setup)
- [Experiment Design](#-experiment-design)
- [Future Roadmap](#-future-roadmap)\n\n---\n\n## 📁 Repository Structure\n\n```text
convolutional-neural-networks/
│
├── CNN Vs ANN/
│   ├── ANN_Vs_CNN.ipynb
│   └── Images/
│
├── Cats_Vs_Dogs_Classification/
│   ├── Cats_Vs_Dogs_Classification.ipynb
│   ├── best_dogs_vs_cats.keras
│   ├── dogs_vs_cats_cnn_final.keras
│   └── dogs_vs_cats_model.png
│
├── LeNet - 5/
│   ├── LeNet_5.ipynb
│   ├── lenet5_mnist.keras
│   ├── lenet5_training_history.csv
│   └── Images/
│
├── Padding and Strides/
│   ├── CNN_Padding_and_Strides.ipynb
│   └── Images/
│
├── Pooling Layers/
│   ├── Pooling_Layers.ipynb
│   └── Images/
│
└── README.md
```

The tree above reflects the repository modules and saved artifacts currently present.\n\n---\n\n## 🎯 Learning Objectives\n\nBy working through this repository, you should be able to explain:

- why convolution is appropriate for spatial data;
- what kernels and filters represent;
- how feature maps are formed;
- how padding changes spatial geometry;
- how stride controls the scanning process;
- how pooling compresses representations;
- how receptive fields grow with depth;
- how CNNs learn hierarchical features;
- how LeNet-5 is structured;
- why CNNs and dense ANNs treat image structure differently;
- how to evaluate classification beyond accuracy;
- how to inspect learned filters and feature maps;
- how to analyze confidence and misclassification;
- how to compare parameter efficiency;
- how to move from MNIST toward natural-image classification.\n\n---\n\n## 🧠 Why CNNs\n\nAn image is not merely a list of unrelated numbers. Nearby pixels usually have meaningful relationships. Edges, curves, textures, strokes, and object parts are spatial structures.

A dense ANN can flatten an image:

```text
28 × 28 → 784
```

A CNN preserves the spatial arrangement:

```text
28 × 28 × 1
```

This enables local connectivity and parameter sharing. A filter can scan across many locations and learn a reusable detector for a visual pattern.

The important idea is not that CNNs are universally superior to every ANN. The important idea is that CNNs encode an architectural bias that is well matched to many image problems.\n\n---\n\n## 🖼️ Image Tensors\n\nCommon image representations are:

```text
Grayscale: H × W × 1
RGB:       H × W × 3
Batch:     B × H × W × C
```

For example:

```text
32 × 224 × 224 × 3
```

means 32 RGB images, each 224×224 pixels.

Understanding shape is one of the most important practical CNN skills. Before debugging a complicated architecture, inspect the tensor dimensions.\n\n---\n\n## 🔲 Convolution\n\nA convolutional layer repeatedly examines local regions of an input.

```text
Input
  ↓
Local patch
  ↓
Multiply by learned weights
  ↓
Sum
  ↓
Add bias
  ↓
Activation
  ↓
Move window
  ↓
Repeat
```

A simplified expression is:

```text
Feature Map = Input ⊛ Kernel + Bias
```

In common deep-learning libraries the operation is technically cross-correlation because the kernel is not flipped, but the conventional term remains convolution.

The useful intuition is that the filter asks the same learned question at many spatial locations: **“How strongly does this pattern appear here?”**\n\n---\n\n## 🧩 Kernels and Filters\n\nA kernel is a small collection of learnable weights. A convolutional layer normally contains many filters, producing many feature maps.

```text
Input
 ├── Filter 1 → Feature Map 1
 ├── Filter 2 → Feature Map 2
 ├── Filter 3 → Feature Map 3
 └── Filter N → Feature Map N
```

The weights are learned through gradient-based optimization. Early filters may become sensitive to edges, orientations, curves, and local contrast. Deeper filters can combine earlier responses into more complex patterns.\n\n---\n\n## 🔥 Feature Maps\n\nA feature map is a spatial record of a filter's response.

```text
Image → learned filter → activation map
```

Strong activation means the learned pattern produced a strong response at that location.

Feature maps therefore provide a bridge between raw pixels and learned representation. The repository explicitly visualizes CNN and LeNet-5 feature maps so the internal transformation is inspectable rather than hidden.\n\n---\n\n## 🧱 Padding\n\nPadding adds values around an input before convolution. Zero padding is the common practical choice.

### VALID

```text
No explicit padding.
Spatial dimensions shrink for typical stride-1 convolution.
```

### SAME

```text
Padding is selected to preserve spatial dimensions for stride 1.
```

Padding changes output dimensions, border participation, and the number of spatial positions processed by later layers.

The dedicated Padding and Strides notebook contains visual experiments around border information, explicit zero padding, spatial-size changes, and feature-map activations.\n\n---\n\n## 🏃 Stride\n\nStride determines how far the kernel moves after each operation.

```text
Stride 1 → dense spatial scanning
Stride 2 → larger jumps and fewer output locations
```

Increasing stride can reduce spatial resolution and computation, but aggressive downsampling may discard useful information.

The repository includes a stride scanner and visualizations showing how stride changes the number of spatial locations.\n\n---\n\n## 🧮 Output Size\n\nFor a single spatial dimension, the common convolution output formula is:

```text
Output = floor((N + 2P − K) / S) + 1
```

where:

- N = input size
- P = padding
- K = kernel size
- S = stride

Example:

```text
N = 28
K = 3
P = 0
S = 1

Output = floor((28 - 3) / 1) + 1
       = 26
```

Therefore a 3×3 VALID convolution with stride 1 maps 28×28 to 26×26.\n\n---\n\n## 🏔️ Pooling\n\nPooling summarizes local regions and usually reduces spatial resolution. The repository contains a dedicated Pooling Layers notebook with visualizations of pooling windows, feature maps, spatial compression, translations, and global average pooling.

Common operations:

- Max pooling
- Average pooling
- Global average pooling

Pooling can reduce computation and provide a degree of local positional tolerance, but it also discards information.\n\n---\n\n## 🥇 Max Pooling\n\nFor a local region:

```text
1 3
2 7
```

max pooling returns:

```text
7
```

The strongest response survives. This is useful when the presence of a feature matters more than every precise activation value.\n\n---\n\n## ➗ Average Pooling\n\nAverage pooling summarizes a local region using its mean.

For:

```text
1 3
2 7
```

we obtain:

```text
(1 + 3 + 2 + 7) / 4 = 3.25
```

Unlike max pooling, it retains average response rather than only the strongest activation.\n\n---\n\n## 🌍 Global Average Pooling\n\nGlobal average pooling summarizes each complete feature map into one value:

```text
H × W × C
   ↓
1 × 1 × C
```

It can reduce classifier parameters substantially compared with flattening a large spatial tensor. The pooling module in this repository includes an explicit global-average-pooling visualization.\n\n---\n\n## 🔭 Receptive Fields\n\nA receptive field describes the portion of the original input that can influence an activation.

Early layers observe small local regions. As layers are stacked, deeper activations depend on larger portions of the original image.

```text
pixels
 ↓
edges
 ↓
curves / textures
 ↓
parts
 ↓
shapes
 ↓
higher-level patterns
```

This growing context is central to hierarchical visual representation.\n\n---\n\n## 🧠 Hierarchical Features\n\nCNNs can be understood as progressively transforming representations:

```text
Raw pixels
    ↓
Simple local patterns
    ↓
Combinations of patterns
    ↓
Shapes and structures
    ↓
Task-relevant representation
    ↓
Prediction
```

This is a conceptual description rather than a claim that every filter maps cleanly to a human semantic label. Visualization helps investigate the learned representation, but should not be treated as a perfect explanation.\n\n---\n\n## ⚡ Activation Functions\n\nA standard CNN activation is ReLU:

```text
ReLU(x) = max(0, x)
```

It introduces nonlinearity. Without nonlinear activations, stacking linear transformations would still produce an overall linear transformation.

Nonlinearity allows a deep network to build increasingly expressive functions from simple operations.\n\n---\n\n## 🧮 Parameter Counting\n\nFor a standard convolutional layer:

```text
Parameters = K × K × C_in × C_out + C_out
```

For a 3×3 convolution with 32 input channels and 64 output filters:

```text
3 × 3 × 32 × 64 + 64 = 18,496
```

Parameter count should be considered together with accuracy, runtime, memory, and generalization.\n\n---\n\n## 🧠 LeNet-5\n\nLeNet-5 is one of the central experiments in this repository. The project uses MNIST to study a classical CNN architecture and inspect both its performance and internal representations.

Artifacts:

```text
LeNet - 5/
├── LeNet_5.ipynb
├── lenet5_mnist.keras
├── lenet5_training_history.csv
└── Images/
```

The image assets cover architecture, MNIST samples, learned C1 filters, 3D filter representations, C3 feature maps, S4 feature maps, early feature extraction, training curves, parameter distributions, confusion matrices, confidence distributions, uncertain predictions, highest-confidence predictions, and misclassified samples.\n\n---\n\n## 🔢 MNIST\n\nMNIST contains grayscale handwritten digits from 0 through 9. It is a useful controlled environment for learning CNN mechanics because the data is relatively small and the task is easy to iterate on.

The repository analyzes MNIST class distributions, pixel intensities, sample images, learned filters, feature maps, class-wise accuracy, confusion matrices, confidence, and errors.\n\n---\n\n## 📈 LeNet-5 Training History\n\nThe repository stores a CSV containing:

```text
accuracy, loss, val_accuracy, val_loss, learning_rate
```

The saved history contains 15 epochs. The recorded training accuracy rises from approximately 89.55% in epoch 1 to approximately 99.95% in epoch 15. Validation accuracy rises from approximately 96.13% to approximately 98.90%.

The recorded learning rate starts near 0.001 and later becomes approximately 0.0005.

These numbers should be read as the results of this saved experiment, not as a universal benchmark for LeNet-5.\n\n---\n\n## ⚖️ CNN vs ANN\n\nThe CNN Vs ANN notebook is a multi-dimensional comparison rather than a one-metric contest. Its artifacts include:

- ANN and CNN confusion matrices;
- learning curves;
- loss across epochs;
- test accuracy comparison;
- trainable parameter comparison;
- training-time comparison;
- per-digit accuracy;
- F1 score by digit;
- prediction-confidence distributions;
- learned feature-space visualizations;
- PCA explained variance;
- misclassified-image grids;
- learned CNN filters and feature maps;
- interactive HTML plots;
- a complete performance dashboard.

The meaningful question is how architectural assumptions influence representation, capacity, optimization, and error behavior under the same experimental setup.\n\n---\n\n## 🐱🐶 Cats vs Dogs\n\nThe repository also contains a natural-image classification project:

```text
Cats_Vs_Dogs_Classification/
├── Cats_Vs_Dogs_Classification.ipynb
├── best_dogs_vs_cats.keras
├── dogs_vs_cats_cnn_final.keras
└── dogs_vs_cats_model.png
```

This extends the learning path beyond MNIST. Natural images introduce color, pose, scale, texture, background, lighting, and intra-class variation.

A typical workflow is:

```text
Images
 ↓
Decode
 ↓
Resize
 ↓
Normalize
 ↓
Batch
 ↓
CNN
 ↓
Binary probability
 ↓
Cat / Dog
```\n\n---\n\n## 📊 Model Evaluation\n\nA complete evaluation stack should include more than accuracy:

```text
Accuracy
   ↓
Precision / Recall / F1
   ↓
Confusion Matrix
   ↓
Confidence Analysis
   ↓
Misclassification Analysis
   ↓
Representation Analysis
   ↓
Robustness Testing
```

Each layer answers a different question. Accuracy summarizes performance; confusion matrices expose class relationships; error analysis explains difficult examples; representation analysis investigates what the network learned internally.\n\n---\n\n## 🧩 Confusion Matrices\n\nA confusion matrix compares true labels with predicted labels. The diagonal contains correct classifications, while off-diagonal entries reveal systematic confusion.

For MNIST, a ten-class confusion matrix can reveal whether particular digit pairs are difficult. A normalized matrix can make class-level error rates easier to compare when class counts differ.\n\n---\n\n## 🎯 Per-Class Analysis\n\nOverall accuracy can hide uneven performance. Per-class accuracy and F1 score reveal whether certain classes behave differently.

The repository includes per-digit accuracy and F1 visualizations for the CNN-vs-ANN analysis and a LeNet-5 accuracy-by-digit visualization.\n\n---\n\n## 🎲 Confidence Analysis\n\nA classifier may output probabilities such as:

```text
[0.01, 0.00, 0.03, 0.94, 0.02, ...]
```

The largest value is commonly treated as the predicted class. But confidence is not identical to correctness.

A strong analysis therefore compares:

```text
high-confidence correct predictions
high-confidence incorrect predictions
low-confidence predictions
confidence by class
```

The repository contains multiple confidence artifacts, including overall and per-digit distributions, output probabilities, highest-confidence predictions, and most uncertain predictions.\n\n---\n\n## 🌐 Feature Space\n\nHidden representations can be projected into a human-readable space. The repository contains static and interactive 3D feature-space visualizations for ANN and CNN representations.

Conceptually:

```text
Raw image space
      ↓
Hidden representation
      ↓
Dimensionality reduction
      ↓
2D / 3D visualization
```

If classes become more separated in a learned representation, that is evidence that the representation may be becoming more discriminative. It is still an exploratory visualization rather than a complete proof of model reasoning.\n\n---\n\n## 🧬 PCA\n\nPrincipal Component Analysis can project high-dimensional feature vectors into a smaller set of components.

```text
High-dimensional representation
            ↓
           PCA
            ↓
       PC1 / PC2 / PC3
            ↓
       Human visualization
```

The repository contains a PCA explained-variance artifact for the ANN-vs-CNN experiment.\n\n---\n\n## 🧯 Error Analysis\n\nMisclassified examples are valuable diagnostic evidence.

For each interesting error, inspect:

```text
True class
Predicted class
Confidence
Image appearance
Similar classes
Possible ambiguity
Relevant feature maps
Possible preprocessing issues
```

The repository contains misclassified-image visualizations for ANN, CNN, and LeNet-5 experiments.\n\n---\n\n## 📈 Training Curves\n\nTraining curves help answer:

- Is the model learning?
- Is loss decreasing?
- Is validation performance improving?
- Is the model beginning to overfit?
- Has the learning rate changed?

A common pattern to watch is:

```text
training accuracy ↑
validation accuracy →
training loss ↓
validation loss → / ↑
```

A widening train/validation gap is a useful signal for investigating generalization.\n\n---\n\n## 🧪 Experiment Design\n\nA controlled experiment changes one major variable at a time.

Weak:

```text
change architecture + optimizer + learning rate + batch size
→ observe accuracy
```

Better:

```text
Baseline
   ↓
Change one factor
   ↓
Keep other conditions controlled
   ↓
Measure
   ↓
Inspect errors
   ↓
Interpret
```

This makes conclusions easier to defend and reproduce.\n\n---\n\n## 🔬 Suggested Experiments\n\n### Kernel size
Compare 3×3 and 5×5 filters.

### Stride
Compare stride 1 and stride 2.

### Padding
Compare SAME and VALID.

### Filter count
Compare 16, 32, 64, and 128 filters.

### Pooling
Compare max pooling and average pooling.

### Depth
Compare shallow and deeper convolutional networks.

### Optimizer
Compare optimizers while holding architecture constant.

### Learning rate
Run a controlled learning-rate sweep.

### Regularization
Compare dropout, weight regularization, and early stopping.

### Robustness
Test translation, rotation, blur, noise, brightness, and contrast changes.\n\n---\n\n## 🛡️ Generalization and Overfitting\n\nA model can approach perfect training accuracy while validation performance stops improving.

Potential responses include:

- better data augmentation;
- regularization;
- dropout;
- early stopping;
- learning-rate schedules;
- reduced model capacity;
- improved data quality.

The correct response depends on the diagnosed failure. Adding every regularization technique at once makes experimentation harder to interpret.\n\n---\n\n## 🧪 Robustness Testing\n\nClean test performance is only one view of a model.

A robustness experiment can follow:

```text
Original image
    ↓
Controlled perturbation
    ↓
Prediction
    ↓
Confidence
    ↓
Accuracy / stability measurement
```

Useful perturbations include translation, rotation, blur, noise, brightness changes, contrast changes, crop, and compression.\n\n---\n\n## 🛠️ Setup\n\n```bash
git clone https://github.com/Maganpreet-Singh/convolutional-neural-networks.git
cd convolutional-neural-networks
python -m venv .venv
.venv\\Scripts\\activate
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

The current repository tree does not include a root-level requirements.txt, so these commands are a practical starting environment rather than a claim about the exact package versions originally used.\n\n---\n\n## ▶️ Running the Notebooks\n\n### Padding and Strides
Open `Padding and Strides/CNN_Padding_and_Strides.ipynb`.

### Pooling Layers
Open `Pooling Layers/Pooling_Layers.ipynb`.

### LeNet-5
Open `LeNet - 5/LeNet_5.ipynb`.

### CNN vs ANN
Open `CNN Vs ANN/ANN_Vs_CNN.ipynb`.

### Cats vs Dogs
Open `Cats_Vs_Dogs_Classification/Cats_Vs_Dogs_Classification.ipynb`.\n\n---\n\n## 🔐 Reproducibility\n\nRecord the following for serious experiments:

```text
Python version
TensorFlow/Keras version
NumPy version
Dataset version
Random seed
Architecture
Optimizer
Learning rate
Batch size
Epochs
Hardware
Training duration
```

Basic seed setup:

```python
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

A seed improves reproducibility but does not guarantee identical results across all hardware and software configurations.\n\n---\n\n## 🚀 Transfer Learning Roadmap\n\nAfter mastering the repository's fundamentals, move toward pretrained models:

```text
LeNet-5
  ↓
VGG / ResNet
  ↓
MobileNet / EfficientNet
  ↓
Transfer Learning
  ↓
Fine-Tuning
  ↓
Deployment
```

A standard transfer-learning workflow is:

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
```\n\n---\n\n## 🌐 Deployment Roadmap\n\nA notebook can become a real application:

```text
Notebook
  ↓
Saved model
  ↓
Inference function
  ↓
API / CLI
  ↓
Web interface
  ↓
Container
  ↓
Cloud deployment
  ↓
Monitoring
```

Potential extensions include a Streamlit classifier, FastAPI endpoint, image-upload application, or educational CNN visualizer.\n\n---\n\n## 🔬 Research Directions\n\nFuture research-style extensions include:

- layer-wise representation analysis;
- confidence calibration;
- robustness benchmarking;
- Grad-CAM and other attribution methods;
- pruning;
- quantization;
- knowledge distillation;
- parameter/latency/memory benchmarking;
- transfer-learning comparisons;
- systematic architecture sweeps.

For each experiment, document the question, hypothesis, setup, baseline, result, error analysis, interpretation, and limitations.\n\n---\n\n## 🏗️ Suggested Future Structure\n\n```text
convolutional-neural-networks/
├── 01_Foundations/
├── 02_Padding_Strides/
├── 03_Pooling/
├── 04_Classic_CNNs/
├── 05_Comparisons/
├── 06_Applications/
├── 07_Transfer_Learning/
├── 08_Explainability/
├── 09_Robustness/
├── 10_Deployment/
├── assets/
├── requirements.txt
├── LICENSE
└── README.md
```

This is a proposed future organization, not the current repository tree.\n\n---\n\n## 🤝 Contribution Guidelines\n\nGood contributions improve correctness, clarity, visualization, reproducibility, experiment quality, or documentation.

For notebooks:

1. Keep cells focused.
2. Explain important tensor transformations.
3. Use meaningful plots.
4. Avoid unnecessary output.
5. Finish experiments with conclusions.

For experiments:

1. Define a question.
2. Establish a baseline.
3. Change one major variable.
4. Measure results.
5. Inspect errors.
6. Document the conclusion and limitations.\n\n---\n\n## ⚖️ License\n\nThe current repository tree does not contain a root-level license file. This README therefore does not assume a license. Add the intended license before distributing the project under specific open-source terms.\n\n---\n\n## 🏁 Final Takeaway\n\nA CNN is not simply a stack of `Conv2D`, `MaxPooling2D`, and `Dense` calls.

It is a learned transformation:

```text
Pixels
  ↓
Tensors
  ↓
Local neighborhoods
  ↓
Learned filters
  ↓
Feature maps
  ↓
Nonlinear representations
  ↓
Spatial compression
  ↓
Hierarchical features
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

The strongest lesson of this repository is therefore:

> **Do not just train the network. Study the network.**

Measure it. Visualize it. Compare it. Inspect its failures. Understand the representation. Then build the next experiment.

The repository provides a clear path from CNN mechanics to applied computer vision: **Padding and Strides → Pooling → LeNet-5 → CNN vs ANN → Cats vs Dogs → Transfer Learning → Robustness → Explainability → Deployment.**

<p align="center"><strong>🧠 Learn the math · 🔬 Run the experiment · 📊 Measure the behavior · 👁️ Visualize the representation · 🧯 Inspect the failure · 🚀 Improve the model</strong></p>

---


# 🔬 Deep-Dive CNN Study Guide

This section turns the repository README into a compact reference manual. It is written so the repository can serve three purposes at once:

1. a project showcase;
2. a personal CNN study notebook;
3. a starting point for future experiments.

---

## 🧠 CNNs in One Mental Model

A convolutional network can be understood as a sequence of transformations:

<pre><code>IMAGE
  │
  ▼
TENSOR
  │
  ▼
LOCAL OPERATIONS
  │
  ▼
FILTER RESPONSES
  │
  ▼
FEATURE MAPS
  │
  ▼
NONLINEAR REPRESENTATIONS
  │
  ▼
SPATIAL DOWNSAMPLING
  │
  ▼
DEEPER FEATURES
  │
  ▼
CLASSIFIER
  │
  ▼
PROBABILITY DISTRIBUTION
  │
  ▼
PREDICTION
  │
  ▼
EVALUATION</code></pre>

Each stage has a distinct job.

The input stage represents the image numerically.

Convolution discovers local spatial patterns.

Activation functions introduce nonlinearity.

Pooling or strided operations reduce spatial resolution.

Later layers compose earlier signals.

The classifier converts the final representation into task-specific outputs.

Evaluation tells you whether that learned representation is useful.

---

# 🧮 The Four Core CNN Design Questions

When designing or reading a CNN, keep asking four questions:

### 1. What does the model see?

This is controlled by:

- input resolution;
- kernel size;
- receptive field;
- stride;
- depth.

### 2. How much can it represent?

This is influenced by:

- number of filters;
- depth;
- dense-layer width;
- bottleneck structure;
- total parameters.

### 3. How much information is preserved?

This depends on:

- padding;
- stride;
- pooling;
- image resizing;
- augmentation;
- compression.

### 4. How expensive is the model?

This depends on:

- spatial resolution;
- number of channels;
- number of filters;
- kernel size;
- parameter count;
- memory access;
- implementation;
- hardware.

Good CNN design is the process of balancing all four.

---

# 🧱 Convolution Layer Anatomy

A convolution layer can be described by:

<pre><code>Input channels
      +
Kernel height
      +
Kernel width
      +
Number of filters
      +
Bias
      +
Stride
      +
Padding
      ↓
Output feature maps</code></pre>

Suppose the input has:

<pre><code>H × W × C_in</code></pre>

and the layer uses:

<pre><code>K × K filters
C_out filters</code></pre>

The output is approximately:

<pre><code>H_out × W_out × C_out</code></pre>

The spatial dimensions depend on padding and stride.

The depth of the output equals the number of filters.

That distinction is critical:

> Kernel size controls local spatial context. Filter count controls output depth.

---

# 🔍 Why Output Depth Equals Number of Filters

Suppose:

<pre><code>Input = 28 × 28 × 1
Filters = 32
Kernel = 3 × 3</code></pre>

The layer learns 32 different detectors.

Therefore the output depth is:

<pre><code>28 × 28 × 32</code></pre>

when spatial dimensions are preserved.

Each channel corresponds to the spatial response of one learned filter.

---

# 🧠 Why a Filter Has Channel Depth

For an RGB image, a 3×3 filter is not merely 3×3.

It is:

<pre><code>3 × 3 × 3</code></pre>

because it must operate across:

- red channel;
- green channel;
- blue channel.

If the input has 64 channels, a standard 3×3 convolution filter has:

<pre><code>3 × 3 × 64</code></pre>

weights, plus a bias if biases are enabled.

---

# 🧮 Exact Convolution Parameter Formula

For a standard convolution:

<pre><code>Parameters =
(K_h × K_w × C_in × C_out)
+
C_out</code></pre>

or:

<pre><code>(K_h × K_w × C_in + 1) × C_out</code></pre>

if one bias exists per output filter.

Example:

<pre><code>Kernel = 3 × 3
Input channels = 16
Filters = 32

Parameters
= (3 × 3 × 16 + 1) × 32
= 4,640</code></pre>

This is useful for architecture comparisons and parameter-budget experiments.

---

# 🧮 Dense Layer Parameter Formula

For a dense layer:

<pre><code>Parameters =
Input units × Output units
+
Output units</code></pre>

Example:

<pre><code>Input units = 3136
Output units = 128

Parameters
= 3136 × 128 + 128
= 401,536</code></pre>

This demonstrates why flattening large feature maps can create large classifier heads.

---

# ⚙️ Convolution vs Dense Connectivity

Consider:

<pre><code>Input feature map = 28 × 28 × 32</code></pre>

A dense layer with 128 units would require:

<pre><code>(28 × 28 × 32) × 128 + 128</code></pre>

which is:

<pre><code>3,211,392 + 128
= 3,211,520 parameters</code></pre>

A small convolutional layer can often provide useful local processing with far fewer weights.

This is the intuition behind convolutional parameter sharing.

---

# 🧠 Local Connectivity

A dense layer can connect one neuron to every input unit.

A convolutional neuron sees only a local region.

For an early feature:

<pre><code>small local region
      ↓
one activation</code></pre>

This reduces the number of direct connections and matches local visual structure.

As layers deepen, the effective receptive field grows.

---

# 🔭 Effective Receptive Field Intuition

Suppose every layer uses:

<pre><code>3 × 3 kernel
stride = 1</code></pre>

The first layer sees a 3×3 region.

A second stacked 3×3 convolution can combine information covering a larger portion of the original image.

A third layer expands the effective context further.

Conceptually:

<pre><code>Layer 1 → local pixels
Layer 2 → local patterns
Layer 3 → combinations of patterns
Layer 4 → larger structures</code></pre>

This is one reason depth can create richer representations without requiring enormous kernels everywhere.

---

# 🏃 Stride as a Sampling Decision

Stride is not simply a technical argument.

It is a sampling decision.

A larger stride means:

<pre><code>fewer spatial samples
       ↓
less spatial detail
       ↓
less computation
</code></pre>

A smaller stride means:

<pre><code>more spatial samples
       ↓
more detail
       ↓
more computation
</code></pre>

This makes stride part of the model's information budget.

---

# 🧱 Padding as a Border Policy

Padding answers:

> What should happen when the filter reaches the image boundary?

Without padding:

- border pixels participate in fewer windows;
- spatial dimensions shrink;
- edge information can disappear quickly.

With padding:

- dimensions can be preserved;
- borders get more opportunities to participate;
- deeper architectures can retain resolution longer.

The dedicated repository notebook visualizes this rather than presenting padding as a one-line API option.

---

# 🏔️ Pooling as Information Compression

Pooling can be understood as a compression operator.

Before:

<pre><code>many spatial values</code></pre>

After:

<pre><code>fewer summary values</code></pre>

The network gains:

- smaller tensors;
- lower computation;
- lower memory;
- potentially more local positional tolerance.

The network loses:

- exact spatial information;
- fine-grained detail;
- some localization precision.

The repository's pooling experiments explicitly visualize the information trade-off.

---

# 🎯 Translation and Local Robustness

Pooling can make a small local shift less disruptive.

Imagine a strong activation moving slightly within a pooling window.

A max-pooling operation can still return a similar value.

This gives a limited form of local translation tolerance.

It should not be interpreted as perfect translation invariance.

That distinction matters.

---

# 🌍 Global Average Pooling

Global average pooling changes the representation from:

<pre><code>H × W × C</code></pre>

to approximately:

<pre><code>C</code></pre>

by averaging each channel across its spatial dimensions.

This means:

> Each channel contributes one global summary value.

Compared with flattening, this can dramatically reduce parameters.

Compared with max pooling, it preserves an average rather than only the strongest local response.

---

# ⚡ Activation Functions in More Detail

A neural network without nonlinearity can collapse multiple linear layers into one linear transformation.

For example:

<pre><code>y = W₂(W₁x)</code></pre>

can be rewritten as:

<pre><code>y = (W₂W₁)x</code></pre>

The activation function breaks that simple linear collapse.

Common activations to study include:

- ReLU;
- sigmoid;
- tanh;
- softmax for multi-class outputs.

Different activations are appropriate for different positions and tasks.

---

# 🧠 Why Softmax Is Used for Multi-Class Classification

For ten mutually exclusive classes, the final layer can produce ten logits.

Softmax converts them into a probability-like distribution whose values sum to one:

<pre><code>p_i = exp(z_i) / Σ exp(z_j)</code></pre>

Conceptually:

<pre><code>logits
 ↓
softmax
 ↓
10 output values
 ↓
sum = 1</code></pre>

The largest output is commonly selected as the predicted class.

---

# 🔢 Binary Classification

Cats vs Dogs is naturally a binary task.

A model may produce one probability:

<pre><code>P(dog)</code></pre>

and infer:

<pre><code>P(cat) = 1 − P(dog)</code></pre>

depending on the chosen output formulation.

This differs from MNIST's ten-way classification.

---

# 🧠 Loss Functions

The training objective tells the model what “wrong” means.

Common classification choices include:

### Binary cross-entropy

Useful for binary classification.

### Categorical cross-entropy

Useful for mutually exclusive multi-class outputs represented as one-hot vectors.

### Sparse categorical cross-entropy

Useful when labels are integer class IDs rather than one-hot vectors.

The loss is not the same thing as accuracy.

A model can improve its loss while accuracy stays unchanged for a period, because probability confidence can change even when the predicted class does not.

---

# 🔄 Forward Propagation

The forward pass:

<pre><code>input
 ↓
layer 1
 ↓
layer 2
 ↓
...
 ↓
output
 ↓
loss</code></pre>

Every layer transforms the representation.

During inference, the forward pass is all that is required.

During training, the forward pass is followed by backpropagation.

---

# 🔁 Backpropagation

The training cycle:

<pre><code>Forward pass
      ↓
Prediction
      ↓
Loss
      ↓
Gradients
      ↓
Weight update
      ↓
Next batch</code></pre>

Backpropagation applies the chain rule to determine how changes in parameters affect the loss.

This is how convolution filters become learned feature detectors.

---

# 🏃 Optimizers

An optimizer uses gradients to update trainable parameters.

Examples:

- SGD;
- Momentum;
- Adam;
- RMSprop.

The optimizer influences:

- convergence speed;
- stability;
- sensitivity to learning rate;
- training dynamics.

For fair experiments, change the optimizer while keeping other major conditions controlled.

---

# 🎚️ Learning Rate

The learning rate controls update magnitude.

Too high:

<pre><code>large updates
↓
possible instability
</code></pre>

Too low:

<pre><code>tiny updates
↓
slow learning
</code></pre>

The ideal range depends on:

- model;
- optimizer;
- dataset;
- batch size;
- normalization;
- initialization.

---

# 🧪 Batch Size

Batch size is the number of training examples used for one optimization step.

Small batches:

- provide noisier gradient estimates;
- can use less memory;
- may introduce different optimization dynamics.

Large batches:

- provide more stable gradient estimates;
- can improve hardware utilization;
- require more memory.

Batch size should be treated as an experimental variable.

---

# 🔥 Epochs

An epoch is one complete pass through the training data.

More epochs do not automatically mean better generalization.

Eventually:

<pre><code>training performance ↑
validation performance →</code></pre>

may indicate diminishing returns or overfitting.

---

# 🛡️ Regularization

Regularization discourages undesirable model behavior or excessive complexity.

Common tools:

- dropout;
- L1 regularization;
- L2 regularization;
- early stopping;
- data augmentation;
- weight decay.

Use them for a reason.

---

# 🌧️ Data Augmentation

Augmentation artificially creates additional training variation.

Possible image transformations:

- translation;
- rotation;
- crop;
- zoom;
- brightness;
- contrast;
- horizontal flip when semantically valid.

The correct transformation depends on the domain.

A transformation that preserves the label is usually more useful than a transformation that creates unrealistic examples.

---

# 🧠 Batch Normalization

Batch normalization standardizes intermediate activations using batch statistics during training.

It can help optimization and influence the distribution of intermediate features.

A future notebook could compare:

<pre><code>CNN
vs
CNN + BatchNorm</code></pre>

while controlling all other settings.

---

# 🧱 Architecture Patterns

A common CNN block is:

<pre><code>Conv
 ↓
Activation
 ↓
Conv
 ↓
Activation
 ↓
Pooling</code></pre>

Another common pattern:

<pre><code>Conv
 ↓
BatchNorm
 ↓
Activation
 ↓
Conv
 ↓
BatchNorm
 ↓
Activation
 ↓
Downsample</code></pre>

Modern architectures can add residual connections, bottlenecks, attention, and depthwise separable convolution.

---

# 🔗 Residual Learning

A residual block introduces a shortcut:

<pre><code>x ───────────────┐
│                 │
▼                 │
Transform         │
│                 │
▼                 │
      Add ◄───────┘
       │
       ▼
      y</code></pre>

Instead of forcing a stack to learn an entirely new mapping, it can learn a residual adjustment.

This idea is central to the ResNet family and is a natural future topic after LeNet-5.

---

# 📱 Depthwise Separable Convolution

Depthwise separable convolution factorizes convolution into:

<pre><code>depthwise convolution
        +
pointwise convolution</code></pre>

This can reduce computation and parameter count compared with standard convolution.

It is particularly important for efficient architectures such as MobileNet.

---

# 🏗️ Classical CNN Progression

A useful historical learning path:

<pre><code>LeNet-5
   ↓
AlexNet
   ↓
VGG
   ↓
Inception
   ↓
ResNet
   ↓
MobileNet
   ↓
EfficientNet</code></pre>

Each family highlights different engineering ideas:

- deeper networks;
- better optimization;
- multi-scale processing;
- residual connections;
- efficiency;
- compound scaling.

---

# 🔍 Model Interpretability Layers

Interpretability can be viewed in levels:

<pre><code>Level 1
Weights / Filters
      ↓
Level 2
Feature Maps
      ↓
Level 3
Embeddings / Feature Space
      ↓
Level 4
Confidence + Errors
      ↓
Level 5
Attribution / Saliency
      ↓
Level 6
Counterfactual Analysis</code></pre>

This repository already contains many Level 1–4 ingredients.

---

# 📊 Confidence vs Correctness Matrix

A useful diagnostic table:

| | Correct | Incorrect |
|---|---|---|
| High confidence | Easy examples | Dangerous overconfidence |
| Low confidence | Ambiguous but correct | Difficult failures |

This is one reason confidence analysis matters.

---

# 🧯 Error Taxonomy

Not all errors are equivalent.

### Type A — Ambiguous sample

The image itself is difficult.

### Type B — Similar classes

Two classes share visual structure.

### Type C — Preprocessing issue

The model sees a representation unlike training data.

### Type D — Distribution shift

The image distribution differs from training data.

### Type E — Model capacity limitation

The architecture may not represent the relevant pattern sufficiently.

### Type F — Optimization issue

The model may not have found a useful solution under the chosen training setup.

A strong error analysis tries to distinguish among these.

---

# 🧪 Data Leakage

A serious ML project must avoid leakage.

Leakage can happen when:

- test examples enter training;
- augmented copies cross splits improperly;
- preprocessing uses future/test information;
- duplicate images appear across datasets.

Always define the split before model fitting.

---

# 🧠 Train / Validation / Test Roles

### Training set

Used to learn parameters.

### Validation set

Used to make development decisions.

### Test set

Used for final unbiased evaluation.

A simple mental model:

<pre><code>TRAIN
learn

VALIDATION
choose / tune

TEST
final evaluation</code></pre>

The test set should not become an informal tuning set.

---

# 📈 Learning Curve Interpretation

### Underfitting

Potential signal:

<pre><code>training score low
validation score low</code></pre>

### Good fit

Potential signal:

<pre><code>training score high
validation score also high
small gap</code></pre>

### Overfitting

Potential signal:

<pre><code>training score very high
validation score significantly lower</code></pre>

These are diagnosis patterns, not mathematical laws.

---

# 🔬 Why Error Analysis Beats Score Chasing

Suppose two models achieve nearly identical overall accuracy.

Model A may fail on:

<pre><code>2 ↔ 7
3 ↔ 8</code></pre>

Model B may fail on:

<pre><code>4 ↔ 9
5 ↔ 8</code></pre>

If your application depends heavily on a specific class, these models are not operationally equivalent.

Therefore:

> Global metrics summarize; error analysis explains.

---

# 🧠 Why Feature-Space Analysis Matters

The classifier only sees the final representation.

Therefore a useful question is:

> What geometry does the network create before classification?

A representation may transform:

<pre><code>messy pixel space
      ↓
less entangled feature space
      ↓
more separable classes</code></pre>

PCA and 3D plots provide one way to inspect this.

---

# 📊 PCA Interpretation

Suppose PCA reports:

<pre><code>PC1 → 40%
PC2 → 20%
PC3 → 10%</code></pre>

Then the first three components explain approximately 70% of the variance.

But that does not mean 70% of “classification information” is guaranteed to be preserved.

Variance and discriminative information are different concepts.

---

# 🧠 Representation vs Prediction

A useful distinction:

### Representation

What information the network encodes.

### Prediction

What decision the classifier makes from that representation.

A model can produce a good representation but a poorly tuned classifier head.

Conversely, a classifier can obtain strong performance without providing an interpretable representation.

This distinction becomes increasingly important in advanced deep learning.

---

# 🧪 Controlled Benchmarking Template

For each architecture, record:

| Category | Record |
|---|---|
| Data | Dataset and split |
| Input | Resolution and channels |
| Preprocessing | Normalization / augmentation |
| Architecture | Full layer list |
| Parameters | Total trainable parameters |
| Optimization | Optimizer + learning rate |
| Batch | Batch size |
| Training | Epochs / scheduler |
| Accuracy | Train / validation / test |
| F1 | Macro / weighted / class-wise |
| Error | Confusion matrix |
| Confidence | Distribution / calibration |
| Runtime | Training + inference |
| Memory | Peak usage when available |

---

# 📦 Model Artifact Checklist

When saving a model, also save the information needed to use it correctly.

Recommended package:

<pre><code>model.keras
preprocessing description
class mapping
input shape
training configuration
evaluation summary</code></pre>

For natural-image projects, preprocessing is especially important.

A model trained on normalized 224×224 images should not be fed arbitrary raw pixels without the expected transformation.

---

# 🧪 Inference Pipeline

A safe inference path:

<pre><code>Raw image
   ↓
Validate file
   ↓
Decode
   ↓
Resize
   ↓
Convert channels
   ↓
Normalize
   ↓
Add batch dimension
   ↓
Model
   ↓
Probability
   ↓
Class mapping
   ↓
Display result</code></pre>

Future deployment work should make this pipeline explicit rather than embedding it invisibly inside a notebook.

---

# 🐱🐶 Cats vs Dogs: Next-Level Experiments

Once the current classifier is understood, try:

### Baseline

Train a small CNN from scratch.

### Augmentation

Add realistic image transformations.

### Transfer learning

Use a pretrained backbone.

### Fine-tuning

Unfreeze selected layers.

### Error analysis

Inspect confusing cat/dog images.

### Robustness

Test brightness, blur, crop, and compression.

### Deployment

Expose the saved model through a simple web application.

---

# 🧠 LeNet-5: Next-Level Experiments

Use the existing LeNet-5 notebook as the baseline.

Then test:

1. fewer filters;
2. more filters;
3. deeper convolution;
4. different pooling;
5. different activations;
6. learning-rate changes;
7. regularization;
8. augmentation;
9. optimizer changes.

For each run, save:

<pre><code>configuration
metrics
history
model
plots
conclusion</code></pre>

This creates a reproducible experiment family.

---

# ⚖️ CNN vs ANN: Fair Comparison Rules

For a defensible comparison:

- use the same dataset;
- use the same split;
- use consistent preprocessing;
- define comparable training budgets;
- document random seeds;
- compare parameter counts;
- compare runtime carefully;
- inspect class-level metrics;
- inspect failure examples.

Changing too many factors can turn an architecture comparison into a collection of unrelated experiments.

---

# 🎓 Learning Milestones

## Milestone 1

You can calculate convolution output sizes manually.

## Milestone 2

You can explain padding and stride without looking at documentation.

## Milestone 3

You can explain why pooling reduces spatial dimensions.

## Milestone 4

You can calculate convolution parameter counts.

## Milestone 5

You can trace a tensor through a CNN layer by layer.

## Milestone 6

You can interpret a confusion matrix.

## Milestone 7

You can explain training vs validation curves.

## Milestone 8

You can inspect learned filters and feature maps.

## Milestone 9

You can diagnose common misclassifications.

## Milestone 10

You can build and evaluate an image classifier without blindly copying a tutorial.

---

# 🧠 CNN Interview Cheat Sheet

<details>
<summary><strong>What is convolution?</strong></summary>

A local weighted operation that applies a learned kernel across spatial positions to produce feature responses.

</details>

<details>
<summary><strong>Why do CNNs share weights?</strong></summary>

The same local detector can be useful at multiple positions, reducing parameter count and exploiting spatial regularity.

</details>

<details>
<summary><strong>What does stride do?</strong></summary>

It controls the movement of the convolution or pooling window and therefore affects spatial sampling and output resolution.

</details>

<details>
<summary><strong>What does padding do?</strong></summary>

It controls border handling and output spatial dimensions.

</details>

<details>
<summary><strong>What does pooling do?</strong></summary>

It summarizes spatial neighborhoods and usually reduces resolution.

</details>

<details>
<summary><strong>Why use multiple filters?</strong></summary>

Different filters can learn different response patterns, producing a richer feature representation.

</details>

<details>
<summary><strong>Why does depth help?</strong></summary>

It enables composition of lower-level features into increasingly complex representations and enlarges effective receptive fields.

</details>

<details>
<summary><strong>Why can a CNN be more parameter-efficient than a dense ANN?</strong></summary>

Because convolution uses local connectivity and reuses the same weights across spatial positions.

</details>

<details>
<summary><strong>Why isn't accuracy enough?</strong></summary>

Because accuracy does not reveal class-specific failures, confidence behavior, parameter cost, or systematic error patterns.

</details>

---

# 📚 Recommended Concept Progression

The repository is best understood in this order:

<pre><code>1. Images as tensors
2. Convolution
3. Kernels and filters
4. Padding
5. Stride
6. Output dimensions
7. Pooling
8. Receptive fields
9. Activations
10. CNN architecture
11. Training
12. Evaluation
13. Feature visualization
14. CNN vs ANN
15. Natural-image classification
16. Transfer learning
17. Explainability
18. Robustness
19. Deployment</code></pre>

---

# 🔬 From Educational Project to Research Project

The project can move toward research quality when every experiment has:

<pre><code>Question
  ↓
Hypothesis
  ↓
Controlled setup
  ↓
Measurement
  ↓
Uncertainty / limitations
  ↓
Interpretation
  ↓
Next hypothesis</code></pre>

Research maturity does not come from using a bigger model.

It comes from asking better questions and controlling the experiment.

---

# 🏗️ Recommended Future File Additions

A useful long-term extension:

<pre><code>experiments/
├── kernel_size/
├── stride/
├── padding/
├── pooling/
├── optimizer/
├── learning_rate/
├── augmentation/
├── robustness/
├── calibration/
└── transfer_learning/

src/
├── data.py
├── models.py
├── evaluation.py
├── visualization.py
└── inference.py

configs/
├── baseline.yaml
├── cnn.yaml
└── transfer_learning.yaml

reports/
├── figures/
└── tables/</code></pre>

This separates reusable engineering code from notebooks and generated outputs.

---

# 📊 Suggested Experiment Tracking Table

| Run | Model | Params | Epochs | LR | Val Acc | Test Acc | F1 | Time | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | Baseline | — | — | — | — | — | — | — | Reference |
| B | Larger CNN | — | — | — | — | — | — | — | More capacity |
| C | Regularized | — | — | — | — | — | — | — | Generalization |
| D | Augmented | — | — | — | — | — | — | — | Robustness |
| E | Transfer | — | — | — | — | — | — | — | Pretrained |

Populate this table from actual experiments rather than estimated values.

---

# 🌟 Portfolio-Level Project Statement

A polished project description for a portfolio can be:

> **A visual, experiment-driven CNN repository exploring convolution, padding, stride, pooling, LeNet-5, feature-map analysis, confidence, error analysis, CNN-vs-ANN comparisons, and natural-image classification with TensorFlow/Keras.**

A stronger technical framing:

> **Built an end-to-end CNN study covering low-level convolution mechanics, classical LeNet-5 modeling, multi-metric ANN/CNN comparison, learned-representation visualization, uncertainty analysis, and applied Cats-vs-Dogs image classification.**

Keep portfolio statements aligned with artifacts actually present in the repository.

---

# 🧭 Final Architecture Checklist

Before building a CNN, answer:

<pre><code>Input size?
      ↓
Number of channels?
      ↓
Kernel size?
      ↓
Number of filters?
      ↓
Padding?
      ↓
Stride?
      ↓
Activation?
      ↓
Downsampling?
      ↓
Receptive field?
      ↓
Parameter count?
      ↓
Classifier head?
      ↓
Loss?
      ↓
Optimizer?
      ↓
Evaluation metrics?
      ↓
Error analysis?</code></pre>

If you can answer every box, you understand the architecture rather than merely writing it.

---

# 🏁 The Core Lesson

The repository is ultimately about a change in mindset.

Beginner mindset:

> “Which layer should I copy?”

Stronger mindset:

> “What representation do I want?”

Advanced mindset:

> “What inductive bias, capacity, sampling strategy, optimization procedure, and evaluation protocol best answer this problem?”

That progression is the real purpose of studying CNNs deeply.

---

<p align="center">
  <strong>🧠 Think in tensors.</strong><br>
  <strong>🔬 Think in experiments.</strong><br>
  <strong>📊 Think in measurements.</strong><br>
  <strong>👁️ Think in representations.</strong><br>
  <strong>🧯 Think in failure modes.</strong><br>
  <strong>🚀 Think in systems.</strong>
</p>
