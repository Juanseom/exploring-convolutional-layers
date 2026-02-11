# Exploring Convolutional Layers

## Problem Description

This project explores the design and behavior of **convolutional neural networks (CNNs)** through systematic experimentation on the Fashion-MNIST dataset.

The goal is to understand how architectural choices in convolutional layers affect performance and learning, comparing them against a non-convolutional baseline. Neural networks are treated here as **architectural components** whose design decisions (kernel size, depth, padding, pooling) have measurable effects on accuracy, efficiency, and generalization.

The project follows a structured approach:
1. Explore and understand the dataset (EDA)
2. Build a baseline model without convolutions
3. Design a CNN with justified architectural decisions
4. Run controlled experiments varying one aspect (kernel size)
5. Interpret the results and explain why convolutions work for image data

---

## Dataset Description

**Dataset:** Fashion-MNIST (from TensorFlow Keras Datasets)

Fashion-MNIST is a collection of **70,000 grayscale images** of clothing items, split into:
- **60,000 training images**
- **10,000 test images**

Each image is **28×28 pixels** with **1 channel** (grayscale), and belongs to one of **10 classes**:

| Label | Class Name |
|-------|------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Bag |
| 8 | Sneaker |
| 9 | Ankle boot |

### Why Fashion-MNIST?

- **Image-based data**: 28×28 grayscale images are the type of data convolutions are designed for.
- **Spatial patterns matter**: Clothing items have shapes, edges, and textures that convolutional filters can detect.
- **Balanced classes**: Each class has exactly 6,000 training and 1,000 test samples — no class imbalance.
- **Fits in memory**: The full dataset runs on any standard laptop.
- **More challenging than MNIST digits**: Fashion items have more visual complexity, making it a better test for CNN experiments.

### Preprocessing Applied

1. **Normalization**: Pixel values divided by 255 (from [0, 255] to [0.0, 1.0])
2. **Reshape for CNN**: Images reshaped from (28, 28) to (28, 28, 1) to include the channel dimension
3. **One-hot encoding**: Labels converted to 10-dimensional vectors for categorical crossentropy loss

---

## Architecture Design

### Baseline Model (Non-Convolutional)

A simple fully connected network that treats each pixel as an independent feature:

```
Input (28×28) → Flatten (784) → Dense(128, relu) → Dense(10, softmax)
```

- **Total parameters**: 101,770
- This model destroys spatial structure by flattening the image into a 1D vector.

### CNN Model

A convolutional network designed with justified architectural choices:

```
Input (28×28×1)
  → Conv2D(32 filters, 3×3, relu, padding='same')
  → MaxPooling2D(2×2)
  → Conv2D(64 filters, 3×3, relu, padding='same')
  → MaxPooling2D(2×2)
  → Flatten
  → Dense(64, relu)
  → Dense(10, softmax)
```

- **Total parameters**: 220,234

### Architecture Diagram

```
┌─────────────────────────────────────┐
│         Input: 28×28×1              │
├─────────────────────────────────────┤
│  Conv2D: 32 filters, 3×3, relu     │  → Output: 28×28×32
│  MaxPooling2D: 2×2                  │  → Output: 14×14×32
├─────────────────────────────────────┤
│  Conv2D: 64 filters, 3×3, relu     │  → Output: 14×14×64
│  MaxPooling2D: 2×2                  │  → Output: 7×7×64
├─────────────────────────────────────┤
│  Flatten                            │  → Output: 3136
│  Dense: 64 neurons, relu            │  → Output: 64
│  Dense: 10 neurons, softmax         │  → Output: 10 (probabilities)
└─────────────────────────────────────┘
```

### Design Justifications

| Decision | Choice | Reason |
|----------|--------|--------|
| Conv layers | 2 | Enough for small 28×28 images. More would risk overfitting. |
| Kernel size | 3×3 | Smallest kernel that captures spatial relationships (a pixel and its 8 neighbors). Computationally efficient. |
| Filters | 32 → 64 | Doubled in the second layer because deeper layers need to represent more complex combinations of features. |
| Padding | 'same' | Preserves spatial dimensions so border information is not lost. |
| Activation | ReLU | Simple, fast, avoids vanishing gradient problem. |
| Pooling | MaxPool 2×2 | Halves spatial dimensions, reduces parameters, and adds tolerance to small spatial shifts. |
| Stride | 1 (Conv2D) | The filter moves one pixel at a time so no information is skipped. |

---

## Experimental Results

### Controlled Experiment: Kernel Size

We varied the kernel size in both Conv2D layers while keeping everything else fixed (same filters, padding, pooling, Dense layers, optimizer, epochs, batch size). This way, any performance difference is only due to the kernel size.

**Kernel sizes tested:** 3×3, 5×5, 7×7

### Results Table

| Model | Kernel Size | Parameters | Test Accuracy | Test Loss |
|-------|-------------|------------|---------------|-----------|
| Baseline (Dense) | — | 101,770 | 88.62% | 0.3264 |
| CNN | 3×3 | 220,234 | **91.72%** | 0.2307 |
| CNN | 5×5 | 253,514 | 90.73% | 0.2623 |
| CNN | 7×7 | 303,434 | 91.35% | 0.2452 |

### Key Observations

- **All CNN models outperformed the baseline** by 2–3 percentage points, confirming that convolutional layers are better suited for image data than fully connected layers.
- **The 3×3 kernel achieved the best accuracy** (91.72%), despite having the fewest parameters among the CNN variants.
- **Larger kernels (5×5, 7×7) did not improve performance** and had significantly more parameters. On 28×28 images, a 7×7 kernel covers 25% of the image width in one step, which may be too coarse for capturing fine details.
- **Trade-off**: Larger kernels increase parameters and computation without a clear accuracy benefit on small images. Two stacked 3×3 convolutions give an effective receptive field of 5×5 with fewer parameters (18 vs 25).

---

## Interpretation

### Why did convolutional layers outperform the baseline?

The baseline model flattens the 28×28 image into a 784-element vector, destroying all spatial structure. It has no concept of "nearby pixels" — every pixel is equally connected to every neuron.

The CNN keeps the 2D structure and uses small filters that scan local regions. This allows it to learn spatial patterns like edges, curves, and textures — exactly the features that make clothing items recognizable.

### What inductive bias does convolution introduce?

1. **Locality**: Each filter looks at a small local region (3×3 pixels). This assumes nearby pixels are more related than distant ones — which is true for images.

2. **Translation equivariance (weight sharing)**: The same filter is applied at every position. A pattern learned in one part of the image (like an edge) can be detected anywhere else. Dense layers would need to learn the same pattern separately for each position.

3. **Hierarchical feature learning**: Stacking convolutional layers builds features from simple to complex. Layer 1 detects edges, Layer 2 combines edges into shapes. This matches how visual patterns work in practice.

### When would convolution NOT be appropriate?

- **Tabular data**: No spatial structure between columns (age, income, etc.)
- **Sequential data with long-range dependencies**: Stock prices, text — RNNs or Transformers are better
- **Graph-structured data**: Social networks, molecules — need Graph Neural Networks
- **Very small non-spatial inputs**: A few sensor readings — Dense layers are sufficient
- **Data where position is arbitrary**: Bag-of-words features — locality assumption does not help

---

## Visualization of Learned Filters and Feature Maps

The notebook includes a section that visualizes what the CNN actually learned:

- **Learned filters (Layer 1)**: The 32 filters of size 3×3 are displayed as small grayscale images. Some resemble edge detectors (horizontal, vertical, diagonal), while others respond to specific textures or intensity changes.

- **Feature maps (Layer 1 — 32 maps)**: Show what each filter "sees" when processing a sample image. Different maps highlight different parts of the image — edges, outlines, flat regions.

- **Feature maps (Layer 2 — 64 maps)**: Smaller (due to MaxPooling) but more abstract. They combine basic features from Layer 1 into more complex patterns, demonstrating the hierarchical nature of CNN feature learning.

---

## Deployment on AWS SageMaker

The notebook was uploaded and executed on **Amazon SageMaker** to train the model in the cloud.

### Process

1. **Uploaded the notebook** to SageMaker Notebook Instance
2. **Executed all cells** including data loading, preprocessing, baseline model, CNN model, and controlled experiments
3. **Trained models in the cloud** using SageMaker's compute resources

### Evidence


---
