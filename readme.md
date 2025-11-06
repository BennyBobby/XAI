# XAI Project in Computer Vision: Understanding CNN Decisions

## Project Context (XAI in Computer Vision)

This project explores and implements **eXplainable Artificial Intelligence (XAI)** techniques applied to Convolutional Neural Networks (CNNs) in **computer vision**. The core goal is to make model predictions transparent by visualising and quantifying which areas of an input image exert the most influence on the model's final decision.

| Method Family | Core Function |
| :--- | :--- |
| **Gradient-Based Methods** | Measure the **sensitivity** of the target score to internal activations or input pixels via backpropagation. |
| **Perturbation-Based Methods** | Measure the **causal impact** on the score by masking, modifying, or removing parts of the input. |


## Implemented Explainability Methods

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

**Type:** **Gradient** and Activation-Based.
**Purpose:** To generate a **low-resolution heatmap** by combining the feature maps from the last convolutional layer with the gradients flowing back from the target class score.

### 2. Deletion Game (Perturbation Method for Evaluation)

**Type:** **Perturbation**-Based.
**Purpose:** To quantify the *goodness* or *fidelity* of a generated saliency map (like Grad-CAM). This is achieved by **systematically removing** the pixels identified as most important and measuring the resulting **drop** in the model's confidence score.

## Setup and Dependencies

The notebooks are implemented in **PyTorch** and utilise ImageNet pre-trained models (VGG16 or ResNet50).

To run the project, you will need the following dependencies (refer to `requirements.txt` for exact versions):

* `torch`
* `torchvision`
* `numpy`
* `matplotlib`
* `opencv-python`
* `Pillow`