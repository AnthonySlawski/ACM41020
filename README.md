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

# Introduction to Python

---
## [Getting Started with Python](https://github.com/AnthonySlawski/ACM41020/blob/main/GettingStartedWithPython.ipynb)

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

# Linear Algebra

---

## [Gaussian Elimination and LU Decomposition](https://github.com/AnthonySlawski/ACM41020/blob/main/LUdecomposition-Solution.ipynb)

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

## [QR Factorization](https://github.com/AnthonySlawski/ACM41020/blob/main/QR-Solution.ipynb)

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

## [Singular Value Decomposition (SVD) and Low-Rank Approximations](https://github.com/AnthonySlawski/ACM41020/blob/main/SVD_LowRankApproximations%20-%20Solution.ipynb)

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

# Understanding Data

---

## [Regression](https://github.com/AnthonySlawski/ACM41020/blob/main/LinearRegression.ipynb)

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

## [Principal Component Analysis (PCA)](https://github.com/AnthonySlawski/ACM41020/blob/main/PrincipalComponentAnalysis.ipynb)

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

# Support Vector Machines

---

## [Support Vector Machines (SVMs)](https://github.com/AnthonySlawski/ACM41020/blob/main/Support%20Vector%20Machines%20-%20Solution.ipynb)

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

# Neural Networks

---

## [Neural Networks](https://github.com/AnthonySlawski/ACM41020/blob/main/Simple%20Neural%20Network%20-%20Solution.ipynb)

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

## [Training a Neural Network with Stochastic Gradient Descent](https://github.com/AnthonySlawski/ACM41020/blob/main/Training%20a%20Neural%20Network%20-%20solution.ipynb)

**Objective**  
Implement a stochastic gradient descent (SGD) learning algorithm to train feedforward neural networks, with applications in binary classification and handwritten digit recognition.

**Methods Used**  
- **Custom Neural Network Implementation**: Developed a `Network` class to manage layers, weights, biases, and training processes.
- **Activation Function**: Utilized the sigmoid function and its derivative for forward propagation and gradient calculations.
- **Stochastic Gradient Descent**: Implemented mini-batch SGD with backpropagation for parameter updates.
- **Visualization**: Plotted decision boundaries for binary classification and visualized network predictions for handwritten digit recognition.

**Key Findings**  
- **Binary Classification**:  
  - Trained a network with two hidden layers (3 neurons each) to classify points as blue or red.  
  - Achieved a clear decision boundary after 10,000 iterations with a learning rate of 0.05.  
- **Handwritten Digit Recognition**:  
  - Constructed a network with one hidden layer (30 neurons) and trained it on the MNIST dataset.  
  - Achieved over 94% accuracy on the test set after 10 epochs.  
- **Training Insights**:  
  - Showcased the role of iterative updates, weight initialization, and learning rate adjustments in optimizing performance.  
  - Highlighted the network's ability to generalize to unseen data.

**Conclusion**  
This notebook demonstrates the effectiveness of stochastic gradient descent in training neural networks for diverse tasks. By applying SGD and backpropagation, the network successfully classified simple binary data and recognized handwritten digits. These results underline the importance of network architecture, parameter tuning, and visualization in achieving optimal performance.

---

## [Improvements to a Neural Network](https://github.com/AnthonySlawski/ACM41020/blob/main/Improved%20Neural%20Network%20-%20Solution.ipynb)

**Objective**  
Enhance a simple feedforward neural network to improve accuracy and training efficiency through cost function selection, regularization, weight initialization, and early stopping.

**Methods Used**  
- **Cost Functions**:
  - **Quadratic Cost**: Measures squared differences between predictions and actual outputs.
  - **Cross-Entropy Cost**: Optimized for classification tasks, improving numerical stability and convergence.
- **Weight Initialization**:
  - **Large Weight Initialization**: Weights initialized with a standard Gaussian distribution.
  - **Default Weight Initialization**: Weights scaled by \(1/\sqrt{n}\) to reduce gradient vanishing/exploding issues.
- **Regularization**:
  - \(L2\) regularization added to the cost function to mitigate overfitting.
- **Early Stopping**:
  - Training halted when evaluation accuracy ceases to improve for a defined number of epochs.
- **Monitoring**:
  - Evaluation of cost and accuracy on both training and validation datasets.

**Key Findings**  
- **Accuracy Improvements**:
  - Switching from quadratic to cross-entropy cost significantly boosted accuracy, particularly for larger networks.
  - Early stopping improved generalization by preventing overtraining.
- **Impact of Regularization**:
  - Regularization effectively reduced overfitting, particularly when training on smaller datasets.
- **Weight Initialization**:
  - Default weight initialization improved convergence rates compared to unscaled weights.
- **Performance Metrics**:
  - A 30-neuron hidden layer achieved ~94.6% accuracy on the MNIST test dataset with cross-entropy cost.
  - A 100-neuron hidden layer increased accuracy to ~96.6% under the same conditions.

**Conclusion**  
This notebook demonstrates the impact of incorporating advanced features into a neural network. Cost function selection, regularization, and weight initialization significantly enhance performance, while early stopping ensures robust generalization. These improvements establish a strong foundation for applying neural networks to complex classification tasks like handwritten digit recognition.


---

## [Neural Networks with TensorFlow](https://github.com/AnthonySlawski/ACM41020/blob/main/TensorFlow.ipynb)

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

## [Convolutional Neural Networks (CNNs)](https://github.com/AnthonySlawski/ACM41020/blob/main/Convolutional%20Neural%20Networks.ipynb)

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

## [Transformers](https://github.com/AnthonySlawski/ACM41020/blob/main/Transformers.ipynb)

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

# Optimization

---

## [Regression and Gradient Descent](https://github.com/AnthonySlawski/ACM41020/blob/main/Regression%20and%20Gradient%20Descent%20-%20Solution.ipynb)

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

## [Unconstrained Optimization](https://github.com/AnthonySlawski/ACM41020/blob/main/Optimisation.ipynb)

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

# Assignments

## [Assignment 1](https://github.com/AnthonySlawski/ACM41020/blob/main/Assignment%201.ipynb)

**Objective**  
Explore matrix decomposition methods, including LU, QR, and SVD, as well as visualization of orthogonal and singular vectors.

**Methods Used**  
- **LU Decomposition**:
  - Implemented Gaussian elimination to decompose matrices into upper-triangular (\(U\)) and lower-triangular (\(L\)) forms.
  - Verified results using elementary matrices and `scipy.linalg.lu`.
- **QR Decomposition**:
  - Computed orthogonal (\(Q\)) and upper-triangular (\(R\)) matrices using:
    - Gram-Schmidt orthogonalization.
    - Householder reflections.
    - NumPy's built-in QR method.
- **Singular Value Decomposition (SVD)**:
  - Derived SVD components (\(U\), \(\Sigma\), \(V^T\)) using eigenvalue decomposition.
  - Verified with NumPy's `np.linalg.svd`.
- **Visualization**:
  - Used 3D plots to visualize:
    - Columns of \(A\) and their transformed orthogonal bases in \(Q\) and \(U\).
    - Singular vectors from SVD.

**Key Findings**  
- LU decomposition successfully factorized \(A\), and results matched across manual and library-based methods.
- QR decomposition via Householder reflections demonstrated numerical stability, while Gram-Schmidt was intuitive but prone to instability.
- SVD highlighted the structural properties of \(A\), with singular vectors aligning along directions of maximum variance.
- Visualizations revealed geometric interpretations of matrix decompositions and the orthogonality of vector bases.

**Conclusion**  
This assignment reinforced fundamental concepts in linear algebra and matrix decomposition. Methods like LU, QR, and SVD were applied successfully, offering insights into numerical stability, orthogonality, and eigenvalue-based decompositions. Visualization enhanced understanding of transformations in vector spaces.

---

## [Assignment 2](https://github.com/AnthonySlawski/ACM41020/blob/main/Assignment%202.ipynb)

**Objective**  
Investigate regression techniques, Principal Component Analysis (PCA), and Support Vector Machines (SVMs) to model data and classify points.

**Methods Used**  
- **Regression**:
  - Linear regression with and without intercept using the normal equations to minimize squared errors.
  - Quadratic regression for fitting a parabola to data.
  - Visualization of regression models with error annotations.
- **PCA**:
  - Centered data and computed covariance matrix to extract principal components.
  - Projected data onto the first principal component and visualized the fit.
  - Compared orthogonal and vertical distances for PCA and regression.
- **SVMs**:
  - Linear SVM classification using dual variables and constraints.
  - Support vector identification and calculation of weights (\(\lambda\)).
  - Mapped 1D data into 2D using trigonometric transformations to classify non-linearly separable data.
- **Visualization**:
  - 2D and 3D visualizations of regression models, PCA fits, and SVM decision boundaries.

**Key Findings**  
- **Regression**:
  - Linear regression without intercept resulted in slightly higher mean squared error than the model with intercept.
  - Quadratic regression minimized squared errors effectively, fitting a parabola to the data points.
- **PCA**:
  - PCA minimized orthogonal distances better than regression, aligning with its goal to maximize variance.
  - Vertical distances were smaller in regression, highlighting the differences in objectives.
- **SVMs**:
  - Identified support vectors and weights for linear classification.
  - Kernel methods successfully separated non-linear data by mapping into higher dimensions.
  - SVM classification models achieved effective separation with clear decision boundaries and margins.
- **Visualization**:
  - Regression and PCA visualizations clarified the geometric interpretations of the fits.
  - SVM plots highlighted decision boundaries, margins, and support vectors for multiple datasets.

**Conclusion**  
This assignment demonstrated regression techniques, PCA, and SVMs as powerful tools for data modeling and classification. PCA excelled in minimizing orthogonal distances, while regression minimized vertical distances. SVMs provided robust classification for both linear and non-linear separable datasets, with kernel methods enabling transformations for complex cases. The visualizations significantly enhanced understanding of the methodologies and their outcomes.

---

## [Assignment 3](https://github.com/AnthonySlawski/ACM41020/blob/main/Assignment%203.ipynb)

**Objective**  
Investigate optimization techniques, specifically gradient descent and gradient descent with momentum, to minimize quadratic functions. Analyze convergence behaviors and compare error reduction rates for both methods.

**Methods Used**  
- **Gradient Descent**:
  - Iterative updates using the gradient of a quadratic function.
  - Step size (\(\alpha\)) determined by the largest eigenvalue of the Hessian matrix.
  - Visualized contours of the quadratic function with optimization steps.
  - Calculated errors and convergence rates.
- **Gradient Descent with Momentum**:
  - Added a momentum term to accelerate convergence.
  - Optimized step size and momentum factor based on eigenvalue properties.
  - Compared convergence rates and error reduction against standard gradient descent.
- **Visualization**:
  - Contour plots of quadratic functions showing optimization paths.
  - Comparison of worst-case direction with optimization steps.
- **Error Analysis**:
  - Computed error norms at each step for both methods.
  - Analyzed error ratios to determine convergence rates.

**Key Findings**  
- **Gradient Descent**:
  - Converges linearly with a consistent error ratio of ~0.75.
  - Slower convergence, particularly in directions corresponding to smaller eigenvalues.
  - Demonstrates predictable reduction in error proportional to step size.
- **Gradient Descent with Momentum**:
  - Accelerates convergence with error ratios decreasing over iterations (~0.57, 0.47, 0.43 in initial steps).
  - Outperforms standard gradient descent by leveraging momentum to "remember" prior steps.
  - More efficient error reduction in poorly conditioned directions.
- **Visualization and Analysis**:
  - Contour plots highlight faster optimization steps with momentum.
  - Momentum-based descent achieves tighter convergence around the minimum within fewer iterations.

**Conclusion**  
Gradient descent and its momentum-enhanced variant are effective for optimizing quadratic functions. While standard gradient descent exhibits predictable but slower convergence, the inclusion of momentum significantly improves performance, particularly for ill-conditioned problems. This assignment underscores the importance of acceleration techniques in iterative optimization methods.

---

## [Assignment 4](https://github.com/AnthonySlawski/ACM41020/blob/main/ML%20Assignment%204.ipynb)

**Objective**  
Develop and analyze an autoencoder architecture and neural network models to explore representation learning and function approximation using the MNIST dataset and periodic functions.

**Methods Used**  
1. **Autoencoder Implementation**:
   - Constructed an encoder-decoder architecture using a custom neural network class.
   - Encoder compresses MNIST images into latent vectors.
   - Decoder reconstructs images from latent vectors.
   - Trained autoencoder with MNIST training data to minimize reconstruction loss.

2. **Latent Space Visualization**:
   - Encoded images into 10-dimensional latent vectors.
   - Analyzed reconstructed images to interpret decoder weights as feature representations.

3. **Reconstruction Performance**:
   - Tested reconstruction on both perfect vectors (one-hot representations) and real MNIST test images.
   - Compared reconstructed images with original images to assess the autoencoder’s effectiveness.

4. **Function Approximation**:
   - Trained two separate neural network models:
     - \( f(x) = \sin(x) \) over \( [0, 2\pi] \).
     - \( g(x) = \frac{1 + \sin(x)}{2} \) over \( [0, 2\pi] \).
   - Models utilized sigmoid activation functions and were trained with MSE loss.
   - Compared model outputs with true functions to evaluate performance.

5. **Extrapolation Analysis**:
   - Evaluated \( g(x) \) model’s extrapolation performance on \( [0, 4\pi] \).
   - Investigated the limitations of neural networks when extending beyond training data.

**Key Findings**  
- **Autoencoder**:
  - Encoder effectively compressed MNIST images into latent vectors.
  - Decoder successfully reconstructed images with distinct features corresponding to digit strokes.
  - Performance improved with deeper networks (e.g., 40 hidden neurons in the encoder and decoder).

- **Function Approximation**:
  - The model approximating \( g(x) = \frac{1 + \sin(x)}{2} \) performed better than \( f(x) = \sin(x) \) due to sigmoid activation compatibility with \( g(x) \)’s range.
  - For \( f(x) \), the network struggled to approximate negative values due to sigmoid output limitations.

- **Extrapolation**:
  - Model performance declined significantly outside the training range (\( [0, 2\pi] \)).
  - Limited training data and the smooth nature of sigmoid activations restricted the model’s ability to capture periodicity.

**Conclusion**  
This assignment highlights the strengths and limitations of neural networks in representation learning and function approximation:
- Autoencoders effectively compress and reconstruct data but require deeper architectures for capturing complex patterns.
- Neural networks excel within training ranges but struggle with extrapolation, especially for periodic functions with limited features or training data.
- Activation function choice significantly impacts model performance, particularly for functions with varying ranges.
