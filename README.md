# ACM-41020  
**Applied Mathematics for Machine Learning**

This repository is dedicated to a collection of machine learning tasks and methodologies I learned from my course ACM41020, focusing on the mathematical foundations and computational techniques used in machine learning. Below is an outline of the key areas covered:

## Neural Networks
- Introduction to Neural Networks
- Activation Functions
- Network Architecture
- Backpropagation
- Universal Approximation Theorem
- Regularization
- Convolutional Neural Networks

## Optimization
- Gradient Descent
- Stochastic Gradient Descent
- Momentum and Adaptive Methods
- Newton’s Method

## Textbooks and Other Resources
- **“Linear Algebra and Learning from Data” by Gilbert Strang**  
  Covers the linear algebra underpinning machine learning.
- **“Deep Learning” by Ian Goodfellow et al.**  
  A state-of-the-art introduction to neural networks.
- **“Mathematics for Machine Learning” by Deisenroth et al.**  
  Explores the mathematical foundations with minimal computation.
- **“Neural Networks and Deep Learning” by Michael Nielsen**  
  An intuitive guide with Python examples.

This repository serves as a resource for understanding the interplay between applied mathematics and machine learning, providing practical applications and theoretical insights.
# Lab Notebooks Overview

This document provides an overview of various lab notebooks, highlighting their objectives, methods, key findings, and conclusions. These notebooks cover fundamental and advanced computational topics, including optimization, machine learning, and numerical methods.

---

## Lab Notebook: Getting Started with Python

**Objective**  
Introduce Python programming basics, emphasizing essential concepts for computational tasks and machine learning applications.

**Methods Used**  
- Interactive coding with Jupyter Notebooks.
- Python data structures (lists, tuples, dictionaries).
- Numerical computations with NumPy arrays.
- Function and class creation for structured programming.
- Data visualization using Matplotlib.

**Key Findings**  
- Established foundational knowledge of Python for computational science.
- Demonstrated efficiency of NumPy’s vectorized operations.
- Showcased Python’s visualization capabilities for data analysis.

**Conclusion**  
This notebook sets the stage for advanced programming and data analysis tasks.

---

## Lab Notebook: Gaussian Elimination and LU Decomposition

**Objective**  
Explore Gaussian elimination and LU decomposition to solve linear systems efficiently and improve numerical stability.

**Methods Used**  
- Gaussian elimination to reduce matrices to upper-triangular form.
- LU decomposition for efficient computation.
- Pivoting strategies for improved numerical stability.
- Validation of decompositions using NumPy and SciPy.

**Key Findings**  
- Highlighted the computational efficiency of \( O(n^3) \) operations.
- Demonstrated the importance of pivoting in ill-conditioned matrices.
- Verified consistency of LU decomposition with original matrices.

**Conclusion**  
Gaussian elimination and LU decomposition are essential tools in numerical mathematics and engineering.

---

## Lab Notebook: QR Factorization

**Objective**  
Analyze methods for QR factorization, including Gram-Schmidt, Householder reflections, and Givens rotations.

**Methods Used**  
- Orthogonalization techniques for numerical stability.
- QR decomposition for eigenvalue computation.
- Applications to dense and sparse matrices.

**Key Findings**  
- Gram-Schmidt is intuitive but less stable for ill-conditioned matrices.
- Householder reflections are robust for dense matrices.
- Givens rotations are ideal for sparse matrices.

**Conclusion**  
QR factorization is versatile, with applications ranging from linear systems to eigenvalue computations.

---

## Lab Notebook: Singular Value Decomposition (SVD) and Low-Rank Approximations

**Objective**  
Investigate SVD for decomposing matrices and its applications in low-rank approximations and image compression.

**Methods Used**  
- SVD for matrix decomposition.
- Low-rank approximations to preserve essential data.
- Image compression and functional approximation.

**Key Findings**  
- SVD effectively identifies essential components of matrices.
- Low-rank approximations are efficient for structured data.
- Demonstrated practical compression of images with simple patterns.

**Conclusion**  
SVD is a powerful tool for dimensionality reduction and data compression.

---

## Lab Notebook: Regression

**Objective**  
Explore regression techniques, including linear regression, logistic regression, and their applications.

**Methods Used**  
- Linear regression with least squares.
- Gradient descent for optimization.
- Logistic regression for classification tasks.

**Key Findings**  
- Highlighted the equivalence of multiple regression solutions.
- Demonstrated the importance of balancing overfitting and underfitting.
- Achieved high accuracy in binary classification tasks.

**Conclusion**  
Regression techniques are versatile tools for data modeling and classification.

---

## Lab Notebook: Principal Component Analysis (PCA)

**Objective**  
Demonstrate PCA for dimensionality reduction and classification tasks.

**Methods Used**  
- Data standardization and SVD for PCA.
- Visualization of reduced-dimensional data.
- Classification of datasets like iris and MNIST.

**Key Findings**  
- PCA effectively reduces dimensions while retaining critical information.
- Demonstrated separability of clusters in reduced dimensions.

**Conclusion**  
PCA is essential for feature extraction, visualization, and classification in high-dimensional datasets.

---

## Lab Notebook: Support Vector Machines (SVMs)

**Objective**  
Explore SVMs for classification tasks using primal and dual optimization methods and the kernel trick.

**Methods Used**  
- Linear and kernel-based SVMs for data classification.
- Visualization of decision boundaries and support vectors.
- Handwriting recognition using the MNIST dataset.

**Key Findings**  
- SVMs are effective for both linear and non-linear classifications.
- Kernel methods enable handling of complex datasets.

**Conclusion**  
SVMs are robust tools for classification tasks, particularly for high-dimensional and complex datasets.

---

## Neural Networks

**Objective**  
Introduce neural networks for image recognition and classification tasks.

**Methods Used**  
- Feedforward and deep neural networks for pattern recognition.
- Backpropagation for parameter optimization.
- Visualization of network weights and hidden layers.

**Key Findings**  
- Neural networks excel at hierarchical feature extraction.
- Deep networks improve performance with sufficient data and layers.

**Conclusion**  
Neural networks are foundational tools for modern machine learning tasks.

---

## Neural Networks with TensorFlow

**Objective**  
Implement and train neural networks using TensorFlow/Keras for handwritten digit classification.

**Methods Used**  
- Data normalization and one-hot encoding.
- Model design with Keras’ Sequential API.
- Training and evaluation on the MNIST dataset.

**Key Findings**  
- Achieved 95.5% accuracy on test data.
- Demonstrated TensorFlow’s simplicity for deep learning tasks.

**Conclusion**  
TensorFlow/Keras enables efficient neural network implementation for practical applications.

---

## Convolutional Neural Networks (CNNs)

**Objective**  
Build a CNN for handwritten digit recognition using TensorFlow/Keras.

**Methods Used**  
- Convolutional and pooling layers for feature extraction.
- Fully connected layers for classification.

**Key Findings**  
- Achieved 99% test accuracy, showcasing CNNs’ power for image tasks.

**Conclusion**  
CNNs are highly effective for tasks requiring spatial feature extraction.

---

## Transformers

**Objective**  
Explore transformer architectures and their applications in natural language processing (NLP).

**Key Topics**  
- Self-attention and multi-head attention mechanisms.
- Applications of transformers in BERT and GPT models.

**Key Findings**  
- Transformers excel in modeling sequential data relationships.
- Enabled advancements in NLP tasks like translation and summarization.

**Conclusion**  
Transformers are revolutionary in NLP, powering state-of-the-art AI systems.

---

## Regression and Gradient Descent

**Objective**  
Explore regression techniques and gradient descent for optimization.

**Key Topics**  
- Linear regression with least squares and gradient descent.
- Convex optimization for regression problems.

**Key Findings**  
- Gradient descent efficiently minimizes objective functions.
- Convex optimization provides exact solutions.

**Conclusion**  
Regression and gradient descent are foundational for predictive modeling and optimization.

---

## Unconstrained Optimization

**Objective**  
Solve optimization problems using gradient descent, momentum-based methods, and stochastic gradient descent (SGD).

**Key Topics**  
- Quadratic and nonlinear optimization problems.
- Faster convergence with momentum-based methods.
- SGD for large-scale problems.

**Key Findings**  
- Momentum accelerates convergence, particularly for ill-conditioned problems.
- SGD balances efficiency and scalability.

**Conclusion**  
Unconstrained optimization techniques are versatile tools for solving mathematical and real-world problems.
