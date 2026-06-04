# Deep Learning

A collection of Deep Learning projects implemented using TensorFlow/Keras and deployed with Streamlit.

## 📖 Overview

This repository demonstrates fundamental Deep Learning concepts through practical projects.

The projects cover:

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Model Training & Evaluation
- Data Visualization
- Interactive Streamlit Applications

---

## 📂 Repository Structure

```text
Deep-Learning/
│
├── ANN/
│   └── ann.py
│
├── CNN/
│   └── cnn.py
│
└── README.md
```

---

# Project 1: Artificial Neural Network (ANN)

## Objective

Compare traditional Machine Learning (Linear Regression) with Deep Learning (Artificial Neural Network) on a study-hours vs marks dataset.

## Features

- Linear Regression Model
- Deep Neural Network Model
- R² Score Comparison
- Marks Prediction Based on Study Hours
- Interactive Streamlit Interface
- Visualization of Actual vs Predicted Values

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Dataset

| Study Hours | Marks |
|------------|--------|
| 1 | 10 |
| 2 | 20 |
| 3 | 35 |
| 4 | 80 |
| 5 | 75 |
| 6 | 90 |
| 7 | 98 |

## ANN Architecture

```text
Input Layer (1 Neuron)
        ↓
Dense Layer (100 ReLU)
        ↓
Dense Layer (100 ReLU)
        ↓
Dense Layer (100 ReLU)
        ↓
Output Layer (1 Neuron)
```

---

# Project 2: Convolutional Neural Network (CNN)

## Objective

Classify handwritten digits using the MNIST dataset.

## Features

- MNIST Digit Classification
- CNN-Based Image Recognition
- Accuracy Evaluation
- Interactive Digit Prediction
- Streamlit Dashboard

## CNN Architecture

```text
Input Image (28×28×1)
        ↓
Conv2D (32 Filters)
        ↓
MaxPooling
        ↓
Conv2D (64 Filters)
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense (128 ReLU)
        ↓
Dense (10 Softmax)
```

## Dataset

MNIST Dataset:

- 60,000 Training Images
- 10,000 Test Images
- Digits 0–9

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Prasadg2023/Deep-Learning.git
```

Move into the project directory:

```bash
cd Deep-Learning
```

Install dependencies:

```bash
pip install tensorflow keras streamlit numpy pandas matplotlib scikit-learn
```

---

## ▶️ Running the Projects

### ANN Project

```bash
streamlit run ANN/ann.py
```

### CNN Project

```bash
streamlit run CNN/cnn.py
```

---

## 📊 Concepts Demonstrated

### Machine Learning

- Linear Regression
- Model Evaluation
- R² Score

### Deep Learning

- Artificial Neural Networks
- Convolutional Neural Networks
- Activation Functions
- Forward Propagation
- Backpropagation
- Image Classification

---

## 🎯 Learning Outcomes

Through these projects, I learned:

- Neural Network Fundamentals
- CNN Architecture Design
- TensorFlow/Keras Implementation
- Model Training & Evaluation
- Data Visualization
- Streamlit Application Development

---

## 🔮 Future Improvements

- Add RNN and LSTM Projects
- Add Transfer Learning Models
- Add Image Upload Prediction
- Add Model Saving & Loading
- Add More Real-World Datasets
- Deploy Applications Online

---

## 👨‍💻 Author

**Prasad Karade**

GitHub: https://github.com/Prasadg2023

---

## ⭐ Support

If you find this repository helpful, please consider giving it a star.
