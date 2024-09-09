import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Sample Data with more entries
np.random.seed(42)
data = {
    'Gender': np.random.choice(['Male', 'Female', 'Non-binary'], 200),
    'Age': np.random.randint(18, 65, 200)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define color palettes
gender_palette = {'Male': '#1f77b4', 'Female': '#ff7f0e', 'Non-binary': '#2ca02c'}
age_palette = sns.color_palette("coolwarm", as_cmap=True)

# Create a figure with custom size
plt.figure(figsize=(14, 7))

# Bar chart for the 'Gender' distribution with annotations
plt.subplot(1, 2, 1)
ax = sns.countplot(x='Gender', data=df, palette=gender_palette, order=df['Gender'].value_counts().index)
plt.title('Gender Distribution', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add annotations on the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', 
                xytext=(0, 10), textcoords='offset points', fontsize=12, color='black')

# Advanced histogram for the 'Age' distribution with KDE and customized bins
plt.subplot(1, 2, 2)
sns.histplot(df['Age'], bins=10, kde=True, color='#ff6361', edgecolor='black', stat="density")
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Density', fontsize=14)

# Customize the style for better readability
sns.despine()
plt.tight_layout()

# Show the plot
plt.show()
