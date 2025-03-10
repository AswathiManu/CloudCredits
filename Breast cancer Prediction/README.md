# Breast Cancer Prediction


## 📌 Project Overview

This project aims to build a Breast Cancer Prediction Model using machine learning techniques. The model classifies tumors as Malignant (Cancerous) or Benign (Non-cancerous) based on extracted features from digitized images of breast mass. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository.

## 📊 Dataset Information

Source: Breast Cancer Wisconsin (Diagnostic) Dataset

Number of Samples: 569

Features: 30 numerical features extracted from digitized images of breast mass.

Target Variable:

M - Malignant (Cancerous)

B - Benign (Non-cancerous)

## 🔍 Attribute Information

ID Number

Diagnosis (M = Malignant, B = Benign)

30 Real-Valued Features Computed for Each Cell Nucleus:

Mean, Standard Error, and Worst (Largest) Value of the following 10 features:

Radius (Mean of distances from center to points on the perimeter)

Texture (Standard deviation of gray-scale values)

Perimeter

Area

Smoothness (Local variation in radius lengths)

Compactness (Perimeter² / Area - 1.0)

Concavity (Severity of concave portions of the contour)

Concave Points (Number of concave portions of the contour)

Symmetry

Fractal Dimension ("Coastline approximation" - 1)

Thus, the dataset contains 10 base features, each with three variations (mean, standard error, and worst value), resulting in 30 features.

## 🛠️ Technologies Used

Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras

Machine Learning Models:

Support Vector Machine (SVM)

Decision Tree

Random Forest

Logistic Regression

K-Nearest Neighbors (KNN)

Neural Networks

## 📈 Model Performance

The models are evaluated based on the following metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix
