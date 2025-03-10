#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = 100


# In[5]:


data = pd.read_csv("Breast_cancer.csv")


# In[6]:


len(data.index), len(data.columns)


# In[7]:


data.shape


# In[8]:


data.head()


# In[9]:


data.tail()


# ### Exploratary Data Analysis

# In[10]:


data.info()


# In[11]:


data.isna()


# In[12]:


data.isna().sum()


# In[13]:


data = data.dropna(axis='columns')


# In[14]:


data.describe(include="O")


# In[15]:


data.diagnosis.value_counts()


# ### To identify dependent and independent values

# In[16]:


diagnosis_unique = data.diagnosis.unique()
diagnosis_unique


# ### Data Visualization

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set Seaborn style and color palette
sns.set_style("whitegrid")
sns.set_palette("coolwarm")


# In[18]:


plt.figure(figsize=(15, 5))

# First subplot: Histogram (Fixed for categorical data)
plt.subplot(1, 2, 1)
data['diagnosis'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title("Counts of Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")

# Second subplot: Seaborn Countplot
plt.subplot(1, 2, 2)
sns.countplot(x='diagnosis', data=data)
plt.title("Diagnosis Count")


# ### Feature Correlation and Class Distribution (Benign vs. Malignant)

# In[19]:


cols = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"]

sns.pairplot(data[cols], hue="diagnosis", palette= "viridis")
plt.show()


# In[20]:


size = len(data['texture_mean'])

area = np.pi * (data['radius_mean']**0.8)  # Bubble size based on radius_mean
colors = data['diagnosis'].map({'B': 'blue', 'M': 'red'})  # Map Benign to blue, Malignant to red

plt.xlabel("texture mean")
plt.ylabel("radius mean") 
plt.scatter(data['texture_mean'], data['radius_mean'], s=area, c=colors, alpha=0.5);


# ### Data Filtering
# 
# Now, we have one categorical feature, so we need to convert it into numeric values using LabelEncoder from sklearn.preprocessing packages

# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


data.head()


# In[23]:


labelencoder_Y = LabelEncoder()
data.diagnosis = labelencoder_Y.fit_transform(data.diagnosis)


# In[24]:


data.head()


# In[25]:


print(data.diagnosis.value_counts())
print("\n", data.diagnosis.value_counts().sum())


# ### Find the correlation between other features, mean features only

# In[26]:


cols = ['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
print(len(cols))
data[cols].corr()


# In[27]:


plt.figure(figsize=(12, 9))

plt.title("Correlation Graph")

cmap = sns.diverging_palette( 1000, 120, as_cmap=True)
sns.heatmap(data[cols].corr(), annot=True, fmt='.1%',  linewidths=.05, cmap=cmap);


# ### Using, Plotly we can show it in interactive graphs like this

# In[28]:


plt.figure(figsize=(15, 10))

fig = px.imshow(
    data[cols].corr(), 
    color_continuous_scale="cividis"
)
fig.show()


# ### Model Implementation

# In[29]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate, cross_val_score

from sklearn.svm import SVC

from sklearn import metrics


# ### Feature Selection

# In[30]:


data.columns


# ### Take the dependent and independent feature for prediction

# In[31]:


prediction_feature = [ "radius_mean",  'perimeter_mean', 'area_mean', 'symmetry_mean', 'compactness_mean', 'concave points_mean']

targeted_feature = 'diagnosis'

len(prediction_feature)


# In[32]:


X = data[prediction_feature]
X


# In[33]:


y = data.diagnosis
y


# ###  Train-test-validation split by 33% and set the 15 fixed records

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)

print(X_train)


# ### Perform Feature Standerd Scaling
# Standardize features by removing the mean and scaling to unit variance
# 
# The standard score of a sample x is calculated as:
# 
# z = (x - u) / s

# In[35]:


# Scale the data to keep all the values in the same magnitude of 0 -1 

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ### Model Building

# In[36]:


def model_building(model, X_train, X_test, y_train, y_test):
    """
    Trains the given model, makes predictions, and evaluates accuracy.

    Parameters:
    model : Machine Learning model instance (e.g., LogisticRegression, RandomForestClassifier)
    X_train : Training features (numpy array or DataFrame)
    X_test : Test features (numpy array or DataFrame)
    y_train : Training target labels (numpy array or Series)
    y_test : Test target labels (numpy array or Series)

    Returns:
    tuple : (train_score, test_accuracy, predictions)
    """
    
    # Fit the model
    model.fit(X_train, y_train)

    # Model score on training data
    train_score = model.score(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy on test data
    test_accuracy = accuracy_score(y_test, predictions)

    return train_score, test_accuracy, predictions


# In[37]:


# Define the models
models_list = {
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=5),
    "DecisionTreeClassifier": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "SVC": SVC()
}

# Train & Evaluate All Models
for model_name, model in models_list.items():
    train_score, test_accuracy, predictions = model_building(model, X_train, X_test, y_train, y_test)
    
    print(f"{model_name}:")
    print(f"   - Training Score  = {train_score:.4f}")  # Model performance on training data
    print(f"   - Test Accuracy   = {test_accuracy:.4f}")  # Model performance on test data
    print("-" * 50)  # Separator for better readability


# ### Define the Confusion Matrix

# In[38]:


def cm_metrix_graph(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=1, linecolor="black")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Example: Compute & Plot Confusion Matrix
y_pred = model.predict(X_test)  # Predicted labels
cm = confusion_matrix(y_test, y_pred)  # Generate confusion matrix

cm_metrix_graph(cm)  # Call function to plot


# ### Plot the Confusion Matrix

# In[39]:


df_prediction = []
confusion_matrixs = []
df_prediction_cols = ['model_name', 'score', 'accuracy_score', "accuracy_percentage"]

for name, model in zip(models_list.keys(), models_list.values()):
    
    (score, accuracy, predictions) = model_building(model, X_train, X_test, y_train, y_test)
    
    print("\n\nClassification Report of '"+ str(name) + "'\n")
    print(classification_report(y_test, predictions))

    df_prediction.append([name, score, accuracy, "{0:.2%}".format(accuracy)])

    # Compute and store confusion matrix
    
    cm = confusion_matrix(y_test, predictions)
    confusion_matrixs.append(cm)
    
    # 🔹 Plot confusion matrix for each model
    
    print(f"Confusion Matrix for {name}")
    cm_metrix_graph(cm)  # Call the function to plot

df_pred = pd.DataFrame(df_prediction, columns=df_prediction_cols)


# In[40]:


df_pred


# ### Understanding K-Fold Cross-Validation in Model Evaluation
# 
# K-Fold Cross-Validation is used to evaluate machine learning models more reliably by splitting the dataset into multiple parts (folds) and training/testing on different subsets.
# 
# ### Key Reasons to Apply K-Fold Cross-Validation:
# More Reliable Performance Estimation → Prevents overfitting by testing the model on different data splits.
# 
# Utilizes the Entire Dataset Efficiently → Each data point gets a chance to be in both training & testing sets.
# 
# Reduces Variability in Results → Helps in obtaining a stable accuracy score.
# 
# Works Well for Small Datasets → Since every observation gets used multiple times for training and testing.
# 

# In[41]:


len(data)


# In[42]:


def cross_val_scoring(model, data, prediction_feature, targeted_feature, k=5):
    """
    Performs K-Fold Cross-Validation and returns accuracy scores.

    Parameters:
    model: Machine Learning model (e.g., LogisticRegression, RandomForestClassifier)
    data: DataFrame containing features and target variable
    prediction_feature: List of feature column names
    targeted_feature: Target variable column name
    k: Number of folds for cross-validation (default is 5)

    Returns:
    dict: { 'train_accuracy': avg_train_score, 'test_accuracy': avg_test_score }
    """

    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    if targeted_feature not in data.columns or not all(feat in data.columns for feat in prediction_feature):
        raise ValueError("Invalid feature or target column names.")

    # Initialize K-Fold cross-validation
    kFold = KFold(n_splits=k, shuffle=True, random_state=42)

    train_scores = []
    test_scores = []

    # Perform cross-validation
    for train_index, test_index in kFold.split(data):
        X_train, X_test = data.iloc[train_index][prediction_feature], data.iloc[test_index][prediction_feature]
        y_train, y_test = data.iloc[train_index][targeted_feature], data.iloc[test_index][targeted_feature]

        # Train model on the current fold
        model.fit(X_train, y_train)

        # Compute accuracy on training and test sets
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = accuracy_score(y_test, model.predict(X_test))

        train_scores.append(train_accuracy)
        test_scores.append(test_accuracy)

    # Compute average scores
    avg_train_score = np.mean(train_scores)
    avg_test_score = np.mean(test_scores)

    print(f"\n📌 Average Train Accuracy: {avg_train_score:.4f}")
    print(f"📌 Average Test Accuracy: {avg_test_score:.4f}")

    return {"train_accuracy": avg_train_score, "test_accuracy": avg_test_score}


# In[43]:


# Loop through all models and apply cross-validation
for name, model in models_list.items():
    print(f"\n🔹 Running Cross-Validation for: {name}")
    
    result = cross_val_scoring(model, data, prediction_feature=["radius_mean", "texture_mean"], targeted_feature="diagnosis", k=5)

    print(f"✅ {name} - Train Accuracy: {result['train_accuracy']:.4f}, Test Accuracy: {result['test_accuracy']:.4f}")


# ### Some of the model are giving perfect scoring. It means sometimes overfitting occurs

# ### Hyperparameter Tuning
# 
# For HyperTunning we can use GridSearchCV to know the best performing parameters
# 
# GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
# 
# The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid

# In[44]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Pick the model
model = DecisionTreeClassifier(random_state=42)

# Tuning Parameters
param_grid = {
    'max_depth': [3, 5, 10, None],  # Controls tree depth
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],  # Minimum samples to split a node
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Minimum samples at leaf node
}

# Implement GridSearchCV with 10-fold cross-validation
gsc = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

# Model Fitting
gsc.fit(X_train, y_train)

# Print the best results
print("\n✅ Best Score: ", gsc.best_score_)
print("\n✅ Best Estimator: ", gsc.best_estimator_)
print("\n✅ Best Parameters: ", gsc.best_params_)


# #### The best hyperparameters for your DecisionTreeClassifier are:
# 
# max_depth = 10 → Limits the depth of the tree to prevent overfitting.
# 
# min_samples_split = 7 → A node must have at least 7 samples before splitting.
# 
# min_samples_leaf = 1 → Each leaf node must contain at least 1 sample.

# ### Applying Hyperparameter tuning for KNeighborsClassifier

# In[45]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Pick the model
model = KNeighborsClassifier()

# Optimized Hyperparameter Grid
param_grid = {
    'n_neighbors': list(range(1, 21, 2)),  # Odd values to prevent ties
    'leaf_size': [1, 5, 10, 20, 30],  # Reduced search space
    'weights': ['uniform', 'distance'],  # Weighting methods
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
}

# Implement GridSearchCV with 10-Fold Cross-Validation
gsc = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)

# Model Fitting
gsc.fit(X_train, y_train)

# Print the best results
print("\n✅ Best Score:", gsc.best_score_)
print("\n✅ Best Estimator:", gsc.best_estimator_)
print("\n✅ Best Parameters:", gsc.best_params_)


# ####  Best score of 91.58% accuracy.
# 
# The average cross-validation accuracy across 10 folds.
# 
# n_neighbors=15: Uses 15 nearest neighbors for classification.
# 
# metric='manhattan': Uses Manhattan distance instead of the default Euclidean.
# 
# leaf_size=1: Optimized for better speed-memory tradeoff in KNN.
# 
# weights='uniform': All neighbors contribute equally to the classification.

# ### Applying Hyperparameter for SVM

# In[56]:


# Pick the model
model = SVC()


# Tunning Params
param_grid = [
              {'C': [1, 10, 100, 1000], 
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000], 
               'gamma': [0.001, 0.0001], 
               'kernel': ['rbf']
              }
]


# Implement GridSearchCV
gsc = GridSearchCV(model, param_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, y_train)

print("\n✅ Best Score is :", gsc.best_score_)

print("\n✅ Best Estimator is :", gsc.best_estimator_ )

print("\n✅ Best Parametes are:", gsc.best_params_)


# #### SVM model achieved a best accuracy of 91.85% using the following hyperparameters:
# 
# ✅ Best Model: SVC(C=10, gamma=0.001, kernel='rbf')
# 
# ✅ Best Accuracy: 91.85%
# 
# ✅ Best Parameters: {C: 10, gamma: 0.001, kernel: 'rbf'}

# ### Applying hyperparameter for Random Forest Classifier

# In[57]:


# Pick the model
model = RandomForestClassifier()


# Tunning Params
random_grid = {'bootstrap': [True, False],
 'max_depth': [40, 50, None], # 10, 20, 30, 60, 70, 100,
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2], # , 4
 'min_samples_split': [2, 5], # , 10
 'n_estimators': [200, 400]} # , 600, 800, 1000, 1200, 1400, 1600, 1800, 2000

# Implement GridSearchCV
gsc = GridSearchCV(model, random_grid, cv=10) # 10 Cross Validation

# Model Fitting
gsc.fit(X_train, y_train)

print("\n✅ Best Score is :", gsc.best_score_)

print("\n✅ Best Estimator is :", gsc.best_estimator_)

print("\n✅ Best Parameters are :", gsc.best_params_)


# ### Conclusion for the Project 🚀
# In this project, we aimed to build an accurate classification model to predict malignant or benign tumors based on medical diagnostic features.
# 
# 1️⃣ Exploratory Data Analysis (EDA) helped us understand feature distributions, correlations, and class imbalances.
# 
# 2️⃣ Feature Engineering & Preprocessing ensured data was clean and suitable for modeling.
# 
# 3️⃣ Model Selection & Evaluation:
# 
# We tested multiple models, including Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine (SVM).
# 
# Hyperparameter tuning using GridSearchCV improved model performance.
# 
# The SVM model with RBF kernel (C=10, gamma=0.001) emerged as the best performer with a cross-validation accuracy of 91.85%.
# 

# ### Final Outcome 🎯
# 
# The SVM model was selected as the final model due to its superior accuracy and generalization ability. It effectively differentiates between benign and malignant tumors, aiding in early cancer detection.

# ### Save the Model

# In[58]:


import pickle

# Save the trained model
with open("svm_model.pkl", "wb") as file:
    pickle.dump(gsc.best_estimator_, file)

print("✅ Model saved successfully!")
