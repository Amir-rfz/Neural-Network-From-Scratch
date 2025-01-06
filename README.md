# Neural Network From Scratch

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [Layer Implementations](#1-layer-implementations)
  - [Forward and Backward Propagation](#2-forward-and-backward-propagation)
  - [Loss Functions and Optimization](#3-loss-functions-and-optimization)
  - [Model Training](#4-model-training)
  - [Model Evaluation](#5-model-evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Contributing](#contributing)

---

## Overview
This project demonstrates the implementation of a simple neural network from scratch in Python. The goal is to gain a fundamental understanding of neural networks by manually implementing layers, activation functions, forward/backward propagation, and training techniques.

The neural network is tested on both regression (California Housing) and classification (MNIST) tasks.

---

## Dataset
### **1. MNIST**
- **Type**: Classification
- **Description**: Handwritten digit images (0-9) of size 28x28 pixels.
- **Goal**: Predict the digit based on pixel intensity.

### **2. California Housing**
- **Type**: Regression
- **Description**: Predict housing prices based on features like location, population, and median income.

---

## Project Workflow

### 1. Layer Implementations
- Implemented layers:
  - **Fully Connected Layer**: Handles the dot product of inputs and weights.
  - **Activation Functions**: Includes ReLU, Sigmoid, and Softmax.
  - **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy Loss.

### 2. Forward and Backward Propagation
- **Forward Propagation**: Compute the output of the network for a given input.
- **Backward Propagation**: Compute gradients for each layer and update weights using gradient descent.

### 3. Loss Functions and Optimization
- Loss functions:
  - **MSE**: For regression tasks (California Housing).
  - **Cross-Entropy Loss**: For classification tasks (MNIST).
- Optimization:
  - **Gradient Descent with Momentum**: Improves convergence speed and avoids local minima.

### 4. Model Training
- The network is trained using the **Stochastic Gradient Descent** algorithm.
- Learning rate and momentum parameters are optimized for both datasets.

### 5. Model Evaluation
- Evaluate the performance of the network using:
  - Accuracy (for classification)
  - Mean Squared Error (for regression)

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Amir-rfz/Neural-Network-From-Scratch.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook ml_models.ipynb
   ```
4. Follow the steps in the notebook to run the data analysis and model evaluations.

---

## Results
- **MNIST Classification**: Achieved 95.08% accuracy after training for 15 epochs.
- **California Housing Regression**: Achieved a mean squared error of 1.493776.

---

## Contributing
Contributions are welcome! Feel free to fork this repository, submit issues, or open pull requests for enhancements or bug fixes.
