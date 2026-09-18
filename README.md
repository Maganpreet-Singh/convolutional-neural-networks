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