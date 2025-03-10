#!/usr/bin/env python
# coding: utf-8

# ### Loading the Libraries

# In[26]:


pip install tensorflow


# In[27]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler


# In[28]:


# LOAD THE DATA
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[29]:


# Split features and labels
Y_train = train["label"]
X_train = train.drop(columns=["label"])


# In[30]:


# Normalize pixel values (scale between 0 and 1)
X_train = X_train / 255.0
X_test = test / 255.0


# In[31]:


# Reshape into 28x28 grayscale images (needed for CNN)
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)


# In[32]:


# Convert labels to one-hot encoding for categorical classification
Y_train = to_categorical(Y_train, num_classes=10)


# In[33]:


# Print shape to verify
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}")


# In[12]:


import matplotlib.pyplot as plt
import numpy as np

# PREVIEW 30 IMAGES
plt.figure(figsize=(15, 4.5))
for i in range(30):  
    plt.subplot(3, 10, i+1)
    plt.imshow(X_train[i].reshape((28,28)), cmap="gray")  # Use "gray" colormap
    plt.axis('off')

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()


# In[34]:


#  CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator(
    rescale=1.0,  # Ensures images aren't double-normalized
    rotation_range=10,  
    zoom_range=0.10,  
    width_shift_range=0.1,  
    height_shift_range=0.1
)


# In[19]:


# PREVIEW AUGMENTED IMAGES

# Select a random starting image
index = np.random.randint(0, len(X_train))
X_train3 = X_train[index].reshape((1,28,28,1))
Y_train3 = Y_train[index].reshape((1,10))

plt.figure(figsize=(15,4.5))

# Generate and display 30 augmented images
for i in range(30):  
    plt.subplot(3, 10, i+1)
    
    # Generate augmented image
    X_train2, Y_train2 = next(iter(datagen.flow(X_train3, Y_train3)))

    # Display image
    plt.imshow(X_train2[0].reshape((28,28)), cmap="gray")
    plt.axis('off')

    # Every 5 images, change the base image
    if i % 5 == 0:  
        index = np.random.randint(0, len(X_train))
        X_train3 = X_train[index].reshape((1,28,28,1))
        Y_train3 = Y_train[index].reshape((1,10))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()


# ### Build 15 Convolutional Neural Networks!

# In[35]:


# BUILD CONVOLUTIONAL NEURAL NETWORKS

nets = 3  # Reduce to 3 models for efficiency
model = [0] * nets

for j in range(nets):
    model[j] = Sequential()

    # Block 1
    model[j].add(Conv2D(32, kernel_size=3, kernel_initializer="he_normal", input_shape=(28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Conv2D(32, kernel_size=3, kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Conv2D(32, kernel_size=5, strides=2, padding='same', kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Dropout(0.3))  # Slightly reduced dropout

    # Block 2
    model[j].add(Conv2D(64, kernel_size=3, kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Conv2D(64, kernel_size=3, kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Conv2D(64, kernel_size=5, strides=2, padding='same', kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Dropout(0.3))  # Slightly reduced dropout

    # Block 3
    model[j].add(Conv2D(128, kernel_size=4, kernel_initializer="he_normal"))
    model[j].add(BatchNormalization())
    model[j].add(Activation('relu'))
    model[j].add(Flatten())
    model[j].add(Dropout(0.3))  # Reduced dropout

    # Output Layer
    model[j].add(Dense(10, activation='softmax'))

    # Compile Model
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print Model Summary
model[0].summary()


# In[36]:


# DECREASE LEARNING RATE EACH EPOCH

# Reduce the number of networks for testing
nets = 3  

# Reduce epochs for quick testing
epochs = 10  

# Decrease learning rate each epoch
annealer = LearningRateScheduler(lambda epoch: 1e-3 * (0.95 ** epoch))

# Train Networks
history = [0] * nets

for j in range(nets):
    # Split data with shuffling
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True)

    print(f"\nTraining CNN Model {j+1}/{nets}...\n")

    history[j] = model[j].fit(
        datagen.flow(X_train2, Y_train2, batch_size=64),
        epochs=epochs,
        steps_per_epoch=len(X_train2) // 64,  # Ensure integer steps
        validation_data=(X_val2, Y_val2),
        callbacks=[annealer],
        verbose=1  # Show training progress
    )

    # Print best accuracy for each model
    print("\n✅ CNN {0:d}: Epochs={1:d}, Best Train Accuracy={2:.5f}, Best Validation Accuracy={3:.5f}\n".format(
        j+1, epochs, max(history[j].history['accuracy']), max(history[j].history['val_accuracy'])
    ))


# ###  Ensembling 15 CNN Models for Final Predictions

# ####  combines predictions from multiple trained CNN models to improve classification accuracy. The ensemble method averages predictions from all models to create a more robust final prediction.
# 

# In[37]:


# Initialize results array for ensemble predictions
results = np.zeros((X_test.shape[0], 10))  

# Collect predictions from each CNN model
for j in range(nets):
    print(f"Predicting with CNN Model {j+1}/{nets}...")
    results += model[j].predict(X_test)  # Summing predictions

# Average the predictions
results /= nets  

# Convert probabilities to final class labels
final_predictions = np.argmax(results, axis=1)

# Create submission file
submission = pd.DataFrame({"ImageId": range(1, len(final_predictions) + 1), "Label": final_predictions})

# Save to CSV
submission.to_csv("MNIST-CNN-ENSEMBLE.csv", index=False)

print("\n✅ Ensemble predictions saved to MNIST-CNN-ENSEMBLE.csv")


# ### Previewing Model Predictions on Test Images
#     visualizes a subset of test images along with their predicted labels from the ensemble CNN model.

# In[38]:


# PREVIEW PREDICTIONS
plt.figure(figsize=(15, 6))

for i in range(40):  
    plt.subplot(4, 10, i + 1)
    plt.imshow(X_test[i].reshape((28, 28)), cmap=plt.cm.binary)  # Ensure correct shape
    plt.title(f"Pred: {final_predictions[i]}", fontsize=10, y=0.85)  # Use final predictions
    plt.axis('off')

# Adjust layout for better readability
plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.show()


# In[ ]:




