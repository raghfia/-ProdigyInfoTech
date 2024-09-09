# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset (assumed to be available locally or via a link)
# You can download it from: https://www.kaggle.com/c/titanic/data
df = sns.load_dataset('titanic')

# Step 1: Overview of the dataset
print("Data Overview:")
print(df.info())  # Information about data types and missing values
print(df.describe())  # Statistical summary of the numerical columns

# Step 2: Handle missing data
# Check for missing values
print("\nMissing Data Count:")
print(df.isnull().sum())

# Fill missing 'age' values with the median age
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing 'embarked' with the most common value (mode)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop rows where 'deck' or 'embark_town' is missing (significant portion missing)
df.drop(columns=['deck', 'embark_town'], inplace=True)

# Step 3: Feature Engineering
# Create a new feature: 'family_size' (combining siblings/spouses and parents/children)
df['family_size'] = df['sibsp'] + df['parch'] + 1  # Including self

# Step 4: Exploratory Data Analysis (EDA)
# Set visual style
sns.set(style="whitegrid")

# Plot 1: Survival Rate by Gender
plt.figure(figsize=(10, 5))
sns.barplot(x="sex", y="survived", data=df, palette="Blues")
plt.title('Survival Rate by Gender')
plt.show()

# Plot 2: Survival Rate by Passenger Class
plt.figure(figsize=(10, 5))
sns.barplot(x="pclass", y="survived", data=df, palette="Blues")
plt.title('Survival Rate by Passenger Class')
plt.show()

# Plot 3: Age Distribution by Survival
plt.figure(figsize=(10, 5))
sns.histplot(df[df['survived'] == 1]['age'], bins=20, kde=True, color='green', label='Survived')
sns.histplot(df[df['survived'] == 0]['age'], bins=20, kde=True, color='red', label='Did Not Survive')
plt.title('Age Distribution by Survival')
plt.legend()
plt.show()

# Plot 4: Family Size and Survival Rate
plt.figure(figsize=(10, 5))
sns.barplot(x="family_size", y="survived", data=df, palette="Blues")
plt.title('Survival Rate by Family Size')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Step 5: Insights & Summary
print("\nExploratory Analysis Summary:")
print("1. Females had a higher survival rate compared to males.")
print("2. First-class passengers had a significantly higher survival rate.")
print("3. Passengers aged between 20 and 40 had more variation in survival rates.")
print("4. Passengers with smaller family sizes (1-2) had better chances of survival compared to those with larger family sizes.")
