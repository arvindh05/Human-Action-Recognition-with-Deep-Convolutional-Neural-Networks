# Human Action Recognition System

This project implements a **Human Action Recognition (HAR)** system that classifies actions within images and identifies if multiple people are present. It utilizes a pre-trained **Xception** model along with deep learning techniques for high-accuracy image classification.

## Project Overview
The main goal is to develop a model capable of recognizing a variety of human actions from images. This system can potentially be applied in areas like security surveillance, sports analytics, and interactive entertainment. 

## Features
- **Human Action Classification**: Detects and classifies actions like 'jumping,' 'cooking,' etc., from images.
- **Person Presence Detection**: Determines if more than one person is present in an image.
- **Indoor-Outdoor Generalization**: Handles a mix of indoor and outdoor scenes to improve generalization in varied environments.

## Dataset
The dataset consists of images labeled with:
- **Class**: Action labels (e.g., "jumping," "cooking").
- **MoreThanOnePerson**: Binary indicator if more than one person is in the image.
- **High-Level Category**: Categories such as 'Social Activities,' 'Sports,' etc.

The training dataset contains 4500 entries, and the test dataset has 3128 entries. Each entry includes the image file path, class label, and additional metadata.

## Technology Stack
- **Programming Language**: Python
- **Libraries and Tools**: 
  - Deep Learning: TensorFlow, Keras, Xception model
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn, Plotly
  - Machine Learning: Scikit-learn, LabelEncoder
- **Hyperparameter Tuning**: Keras Tuner with Random Search for learning rate and regularization tuning.

## Model Architecture
The model leverages **Xception** as the base model for feature extraction, followed by custom layers for classification. 
- **Loss Functions**:
  - Action Classification: Categorical Crossentropy
  - Person Presence Detection: Binary Crossentropy
- **Metrics**:
  - Action Classification: Accuracy
  - Person Detection: Binary Accuracy
- **Optimizer**: Adam (adaptive learning rates and momentum for stability)

## Model Training and Evaluation
- **Early Stopping** and **TensorBoard** for monitoring training and avoiding overfitting.
- **Random Search Tuning** for optimizing hyperparameters such as learning rate and L2 regularization.
- **Evaluation Metrics**:
  - Accuracy and F1-score to assess model performance on both the training and validation datasets.

## Results
- **Training Accuracy**: Reached over 90% for action classification.
- **Validation Trends**: Model shows slight overfitting, with consistent validation accuracy across epochs.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arvindh05/Human-Action-Recognition-with-Deep-Convolutional-Neural-Networks.git
