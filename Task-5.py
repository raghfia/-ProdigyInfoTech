# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import geopandas as gpd
from datetime import datetime

# Step 1: Load the dataset (Example: Traffic Accident Dataset)
# You can use publicly available datasets such as the US Accident Dataset from Kaggle: https://www.kaggle.com/sobhanmoosavi/us-accidents
url = "path_to_your_accident_dataset.csv"
df = pd.read_csv(url)

# Step 2: Data Preprocessing
# Convert date-time columns to proper format
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['End_Time'] = pd.to_datetime(df['End_Time'])

# Extract useful features from datetime columns
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Hour'] = df['Start_Time'].dt.hour
df['Day_of_Week'] = df['Start_Time'].dt.dayofweek  # Monday=0, Sunday=6

# Step 3: Analyze Patterns

# Plot accidents by time of day
plt.figure(figsize=(10, 6))
sns.histplot(df['Hour'], bins=24, kde=False, color='blue')
plt.title('Accidents by Time of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.show()

# Plot accidents by weather condition
plt.figure(figsize=(10, 6))
sns.countplot(y='Weather_Condition', data=df, order=df['Weather_Condition'].value_counts().index[:10], palette='coolwarm')
plt.title('Top 10 Weather Conditions in Accidents')
plt.xlabel('Count')
plt.ylabel('Weather Condition')
plt.grid(True)
plt.show()

# Plot accidents by road condition
plt.figure(figsize=(10, 6))
sns.countplot(y='Road_Condition', data=df, order=df['Road_Condition'].value_counts().index[:10], palette='coolwarm')
plt.title('Top 10 Road Conditions in Accidents')
plt.xlabel('Count')
plt.ylabel('Road Condition')
plt.grid(True)
plt.show()

# Plot accidents by day of the week
plt.figure(figsize=(10, 6))
sns.countplot(x='Day_of_Week', data=df, palette='coolwarm')
plt.title('Accidents by Day of the Week')
plt.xlabel('Day of the Week (0=Monday, 6=Sunday)')
plt.ylabel('Number of Accidents')
plt.grid(True)
plt.show()

# Step 4: Visualize Accident Hotspots using Folium

# Filter rows with valid latitude and longitude
df_geo = df.dropna(subset=['Start_Lat', 'Start_Lng'])

# Create a map centered at an approximate location
m = folium.Map(location=[df_geo['Start_Lat'].mean(), df_geo['Start_Lng'].mean()], zoom_start=12)

# Add a heatmap to the map
heat_data = [[row['Start_Lat'], row['Start_Lng']] for index, row in df_geo.iterrows()]
HeatMap(heat_data).add_to(m)

# Display the map
m

# Step 5: Analyzing Contributing Factors
# Create a correlation matrix for numerical columns
plt.figure(figsize=(12, 6))
corr_matrix = df[['Severity', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Accident Severity and Weather Conditions')
plt.show()

# Analyze accident severity vs road conditions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Road_Condition', y='Severity', data=df)
plt.title('Accident Severity by Road Condition')
plt.xticks(rotation=90)
plt.show()

# Analyze accident severity vs weather conditions
plt.figure(figsize=(10, 6))
sns.boxplot(x='Weather_Condition', y='Severity', data=df)
plt.title('Accident Severity by Weather Condition')
plt.xticks(rotation=90)
plt.show()
