# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset (downloaded locally or from a URL)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv"
df = pd.read_csv(url, sep=';')

# Step 1: Data Exploration
print("Dataset Overview:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())  # Check for missing values (none expected in this dataset)

# Step 2: Data Preprocessing
# Convert categorical columns using one-hot encoding for non-binary categories
df_encoded = pd.get_dummies(df, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'], drop_first=True)

# Convert binary categorical columns (e.g., 'yes'/'no' to 0/1)
df_encoded['default'] = df_encoded['default'].map({'yes': 1, 'no': 0})
df_encoded['housing'] = df_encoded['housing'].map({'yes': 1, 'no': 0})
df_encoded['loan'] = df_encoded['loan'].map({'yes': 1, 'no': 0})
df_encoded['y'] = df_encoded['y'].map({'yes': 1, 'no': 0})  # Target variable

# Step 3: Split the data into features and target variable
X = df_encoded.drop('y', axis=1)  # Features
y = df_encoded['y']  # Target (whether the customer subscribed)

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 6: Visualize the Decision Tree (Optional)
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.show()
