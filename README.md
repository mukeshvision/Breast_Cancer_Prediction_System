# Breast Cancer Prediction using Neural Networks (Deep Learning)

This project implements a deep learning-based binary classification model to predict whether a breast tumor is **benign** or **malignant** using a Neural Network. Built using TensorFlow/Keras, the system is trained on a real medical dataset and includes an interactive **predictive system** to test new patient data.

---

## Project Objective

To develop an accurate and reliable deep learning model that can help medical professionals in early detection and classification of breast cancer. The system should not only classify tumors correctly but also provide a user-interactive predictive interface for real-time diagnosis support.

---

## Technologies Used

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn (EDA & Visualization)
- TensorFlow & Keras (Neural Network Model)
- Scikit-learn (Preprocessing & Evaluation)

---

## Dataset Overview

- **Source**: UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- **Total Records**: 569
- **Features**:
  - Mean radius, texture, perimeter, area, smoothness, etc. (30 features)
- **Target**:
  - `M`: Malignant (Cancerous)
  - `B`: Benign (Non-cancerous)

---

## Workflow

### 1. Data Preprocessing
- Removed irrelevant columns (e.g., ID)
- Label encoding (`M` = 1, `B` = 0)
- Feature scaling using MinMaxScaler
- Data split into training and test sets (80/20)

### 2. Model Architecture
- Input Layer: 30 features
- Hidden Layers: 2 dense layers with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation
- Loss: Binary Crossentropy
- Optimizer: Adam

### 3. Model Evaluation
- Accuracy achieved: **~98.2%**
- Evaluated using:
  - Confusion Matrix
  - Classification Report
  - Accuracy & Loss plots

---

## Predictive System

An interactive predictive function is implemented at the end of the notebook where users can:
- Input 30 feature values (from a test case or new sample)
- Get instant prediction: **"The Breast Cancer is Malignant"** or **"The Breast Cancer is Benign"**

---

## Results

- ✅ High accuracy with low false positives
- ✅ Strong model generalization on unseen data
- ✅ Easy-to-use predictive function for new inputs
