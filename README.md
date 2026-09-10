# Residual Networks (ResNet-50) for Image Classification

A from-scratch implementation of a **ResNet-50** architecture using TensorFlow/Keras, built to understand how residual learning, skip connections, identity blocks, and convolutional blocks enable the training of deep convolutional neural networks.

> This project was developed as part of my deep learning learning journey, with educational guidance and inspiration from Professor Andrew Ng's deep learning coursework.

## 📌 Overview

Very deep neural networks can represent increasingly complex functions, but training them becomes difficult because of problems such as **vanishing gradients**.

Residual Networks (ResNets), introduced by He et al., address this challenge through **shortcut/skip connections**. Instead of forcing every block to learn a completely new transformation, the network can learn a residual mapping while directly passing information through the shortcut.

In this project, I implemented the core components of ResNet-50 and trained the resulting model on the **SIGNS dataset**, a six-class hand-sign image classification dataset.

## 🎯 Project Goals

- Understand why very deep neural networks can be difficult to train.
- Implement residual learning using skip connections.
- Build both major ResNet block types:
  - Identity block
  - Convolutional block
- Assemble those blocks into a **50-layer ResNet architecture**.
- Train the model for multi-class image classification.
- Evaluate the model on unseen test images.
- Test the trained network on an individual image.

## 🧠 Key Concepts

### The Vanishing Gradient Problem

As networks become deeper, gradients can become extremely small during backpropagation. This can make the early layers learn very slowly.

Residual connections provide a direct path through the network, helping information and gradients flow through deep architectures.

### Skip Connections

A residual block contains:

```text
                ┌──────────────────────────────┐
                │                              │
Input ──────────┴──────► Main Path ───────► Add ───► ReLU
  │                                           ▲
  └────────────── Shortcut / Skip ────────────┘
```

The output combines the transformed main path with the shortcut:

```text
Output = ReLU(Main Path + Shortcut)
```

This structure makes it easier for a block to learn an identity mapping when that is useful.

## 🧱 ResNet Building Blocks

### 1. Identity Block

The identity block is used when the input and output dimensions match.

The main path follows:

```text
CONV 1×1 → BatchNorm → ReLU
        ↓
CONV f×f → BatchNorm → ReLU
        ↓
CONV 1×1 → BatchNorm
```

The original input is then added back:

```text
Main Path + Shortcut → ReLU
```

The implementation uses:

- 1×1 convolution
- f×f convolution
- 1×1 convolution
- Batch Normalization
- ReLU activation
- Add layer for the shortcut connection

### 2. Convolutional Block

The convolutional block is used when the input and output dimensions need to change.

Unlike the identity block, the shortcut path contains a **1×1 convolution** and Batch Normalization so that its dimensions match the main path before addition.

The stride can also be used to reduce spatial dimensions.

## 🏗️ ResNet-50 Architecture

The implemented network follows the classic stage-based ResNet-50 structure:

```text
Input (64×64×3)
        │
Zero Padding
        │
7×7 Conv, 64 filters, stride 2
        │
BatchNorm → ReLU
        │
3×3 MaxPool, stride 2
        │
        ▼
Stage 2
Conv Block [64, 64, 256]
+ 2 Identity Blocks
        │
        ▼
Stage 3
Conv Block [128, 128, 512]
+ 3 Identity Blocks
        │
        ▼
Stage 4
Conv Block [256, 256, 1024]
+ 5 Identity Blocks
        │
        ▼
Stage 5
Conv Block [512, 512, 2048]
+ 2 Identity Blocks
        │
        ▼
Average Pooling
        │
Flatten
        │
Dense → 6 classes
        │
Softmax
```

The notebook implements the architecture using the Keras Functional API.

### Architecture Summary

| Component | Configuration |
|---|---|
| Input | `(64, 64, 3)` |
| Initial Conv | 64 filters, 7×7, stride 2 |
| Max Pool | 3×3, stride 2 |
| Stage 2 | [64, 64, 256] + 2 identity blocks |
| Stage 3 | [128, 128, 512] + 3 identity blocks |
| Stage 4 | [256, 256, 1024] + 5 identity blocks |
| Stage 5 | [512, 512, 2048] + 2 identity blocks |
| Final Pooling | Average Pooling |
| Classifier | Dense layer |
| Output | 6 classes with Softmax |

## 📊 Dataset

The project uses the **SIGNS dataset**.

The notebook reports:

- **1,080 training examples**
- **120 test examples**
- Image shape: **64 × 64 × 3**
- **6 target classes**

The image vectors are normalized by dividing pixel values by `255`.

Labels are converted into one-hot encoded matrices for categorical classification.

```python
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
```

## ⚙️ Training

The model is compiled with:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00015)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Training configuration:

- Epochs: **10**
- Batch size: **32**
- Optimizer: **Adam**
- Learning rate: **0.00015**
- Loss: **Categorical Cross-Entropy**
- Metric: **Accuracy**

## 📈 Results

The model trained for 10 epochs on the SIGNS training set.

### Training Progress

The recorded training run reached:

- **Epoch 1:** 31.48% accuracy, loss 1.8572
- **Epoch 5:** 88.06% accuracy, loss 0.3193
- **Epoch 10:** **94.17% accuracy**, loss 0.1647

The training loss generally decreased while accuracy increased throughout training.

### Test Performance

On the 120-example test set, the recorded run achieved:

| Metric | Result |
|---|---:|
| Test Loss | **1.0717** |
| Test Accuracy | **75.83%** |

The notebook also evaluates a separately provided pretrained ResNet-50 model trained for more iterations. That model achieved:

| Metric | Result |
|---|---:|
| Test Loss | **0.1596** |
| Test Accuracy | **95.00%** |

The two results should not be treated as the same training run: the first is the model trained in the notebook for 10 epochs, while the second comes from the separately loaded `resnet50.h5` model.

## 🔍 Testing on a Custom Image

The notebook includes an optional workflow for testing the trained model on a user's own image.

The image is:

1. Loaded from the `images/` directory.
2. Resized to `64 × 64`.
3. Converted to an array.
4. Expanded to a batch dimension.
5. Normalized by dividing by `255`.
6. Passed through the pretrained model.
7. Classified using the class with the highest predicted probability.

Example:

```python
img_path = 'images/my_image.jpg'
```

The notebook's example prediction produced:

```text
Class: 2
```

It also demonstrates an important practical lesson: high test accuracy on a benchmark dataset does not guarantee equally strong performance on images captured under different conditions. Image shape, lighting, and preprocessing can create a distribution shift between the training data and real-world images.

## 🧪 Model Verification

The implementation was checked using the provided testing utilities.

The notebook reports:

```text
All tests passed!
```

for both:

- `identity_block`
- `convolutional_block`

The completed ResNet-50 implementation was also compared against the provided reference summary and passed the supplied verification.

## 🛠️ Technologies

- Python
- TensorFlow
- Keras
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook

## 📁 Project Structure

```text
resnet50-image-classification/
│
├── datasets/
├── images/
├── models/
├── .ipynb_checkpoints/
├── __pycache__/
│
├── Residual_Networks.ipynb
├── outputs.py
├── public_tests.py
├── resnets_utils.py
├── test_utils.py
├── resnet50.h5
├── .gitignore
└── LICENSE
```

> Generated/cache directories and model files are excluded from Git tracking according to the project's `.gitignore`.

The repository's `.gitignore` includes:

```text
.ipynb_checkpoints/
__pycache__/
models/
resnet50.h5
Thumbs.db
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd resnet50-image-classification
```

### 2. Install dependencies

Install the main packages used by the notebook:

```bash
pip install tensorflow numpy scipy matplotlib jupyter
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
Residual_Networks.ipynb
```

### 4. Run the notebook

Run the notebook from top to bottom to:

- explore the vanishing-gradient problem,
- implement the identity block,
- implement the convolutional block,
- construct ResNet-50,
- train the network,
- evaluate it on the test set,
- and optionally test a custom image.

## 📚 What I Learned

This project helped me move beyond simply using a CNN and understand how a deep residual architecture is constructed internally.

Key lessons include:

- Why increasing network depth can make optimization harder.
- How skip connections help deep networks learn effectively.
- The difference between identity and convolutional residual blocks.
- How 1×1 convolutions can control channel dimensions.
- How stride changes spatial dimensions.
- How Batch Normalization fits into residual blocks.
- How the Keras Functional API can represent branching architectures.
- How a large training accuracy does not necessarily guarantee equivalent performance on images from a different distribution.

## 🌍 Real-World Consideration: Distribution Shift

One of the most useful lessons from the project is that benchmark performance is not the whole story.

The notebook demonstrates that a model trained and evaluated on one image distribution can behave differently on personally captured images because of differences in:

- lighting,
- image composition,
- image shape,
- preprocessing,
- and overall data distribution.

This is an important consideration when moving from a controlled dataset to real-world computer vision applications.

## 📖 References

This project is based on the Residual Network architecture introduced in:

**Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.**  
*Deep Residual Learning for Image Recognition*, 2015.

The notebook also acknowledges the implementation structure and reference code from **François Chollet's** deep-learning-models repository.

This project was also part of my learning journey inspired and guided by the educational work of **Professor Andrew Ng**.

## 🙏 Acknowledgments

A special thanks to **Professor Andrew Ng** and the **DeepLearning.AI** team for their educational resources and deep learning curriculum.

Their teaching has been an important part of my journey toward understanding neural networks and computer vision.

## 🤝 Contributing

This repository is primarily a learning project, but suggestions, improvements, and discussions are welcome.

If you notice an issue or have an idea for improving the implementation or documentation, feel free to open an issue or submit a pull request.

## 👤 Author

**Qais Al-Tloa**

- 📧 [Email](mailto:qaisaltloa1@gmail.com)
- 🔗 [LinkedIn](https://www.linkedin.com/in/qais-al-tloa-911427330/)

## 📄 License

A `LICENSE` file is included in the repository. Please refer to that file for the exact license terms.
