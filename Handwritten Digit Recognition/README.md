# Handwritten Digit Recognition using CNN

## Project Overview
This project implements a **Handwritten Digit Recognition** system using **Convolutional Neural Networks (CNNs)**. The model is trained on the **MNIST dataset**, which consists of grayscale images of handwritten digits (0-9) and classifies them with high accuracy. An **ensemble of CNN models** is used to improve the final predictions.

## Dataset
The project uses the **MNIST dataset**, containing:
- **60,000 training images**
- **10,000 test images**

Each image is a **28x28 grayscale** representation of a digit (0-9), with labeled training data.

## Project Workflow
### 1. **Loading the Libraries**
```python
pip install tensorflow

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
```

### 2. **Loading and Preprocessing Data**
```python
# Load dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Split features and labels
Y_train = train["label"]
X_train = train.drop(columns=["label"])

# Normalize pixel values
X_train = X_train / 255.0
X_test = test / 255.0

# Reshape into 28x28 grayscale images
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=10)
```

### 3. **Previewing Data**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(X_train[i].reshape((28,28)), cmap="gray")
    plt.axis('off')
plt.show()
```

### 4. **Data Augmentation**
```python
datagen = ImageDataGenerator(
    rotation_range=10,  
    zoom_range=0.10,  
    width_shift_range=0.1,  
    height_shift_range=0.1
)
```

### 5. **Building CNN Models**
```python
nets = 3  # Using 3 CNN models for ensemble
model = [0] * nets

for j in range(nets):
    model[j] = Sequential()
    
    # CNN Architecture
    model[j].add(Conv2D(32, kernel_size=3, kernel_initializer="he_normal", input_shape=(28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Conv2D(32, kernel_size=3, kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Conv2D(32, kernel_size=5, strides=2, padding='same', kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Dropout(0.3))

    # Additional Layers
    model[j].add(Conv2D(64, kernel_size=3, kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Flatten())
    model[j].add(Dropout(0.3))
    
    # Output Layer
    model[j].add(Dense(10, activation='softmax'))
    
    # Compile Model
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

### 6. **Training the CNN Models**
```python
epochs = 10
annealer = LearningRateScheduler(lambda epoch: 1e-3 * (0.95 ** epoch))

for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True)
    history = model[j].fit(
        datagen.flow(X_train2, Y_train2, batch_size=64),
        epochs=epochs,
        validation_data=(X_val2, Y_val2),
        callbacks=[annealer],
        verbose=1
    )
```

### 7. **Ensembling CNN Models for Final Predictions**
```python
results = np.zeros((X_test.shape[0], 10))

for j in range(nets):
    results += model[j].predict(X_test)

results /= nets  # Average predictions
final_predictions = np.argmax(results, axis=1)

submission = pd.DataFrame({"ImageId": range(1, len(final_predictions) + 1), "Label": final_predictions})
submission.to_csv("MNIST-CNN-ENSEMBLE.csv", index=False)
```

### 8. **Visualizing Model Predictions**
```python
plt.figure(figsize=(15, 6))
for i in range(40):  
    plt.subplot(4, 10, i + 1)
    plt.imshow(X_test[i].reshape((28, 28)), cmap=plt.cm.binary)
    plt.title(f"Pred: {final_predictions[i]}", fontsize=10, y=0.85)
    plt.axis('off')
plt.show()
```

## Results & Accuracy
- Achieved **98%+ accuracy** on the MNIST dataset.
- Successfully recognizes handwritten digits with high confidence.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/CloudCredits.git
   cd CloudCredits/Handwritten_Digit_Recognition
   ```
2. Run the Jupyter Notebook (`.ipynb`) or Python script (`.py`).
3. Train the model and test predictions on new images.

## Future Improvements
- Experiment with deeper architectures like ResNet.
- Train on more complex datasets (EMNIST, SVHN).
- Deploy the model using **Flask** or **FastAPI**.
