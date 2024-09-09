# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Sample social media data (e.g., tweets or reviews)
data = {'text': [
    "I absolutely love this new phone! The camera quality is amazing!",
    "I'm so frustrated with this service, the support team is awful.",
    "This product is okay, not the best but gets the job done.",
    "I had an amazing experience at the store, customer service was excellent!",
    "Not satisfied with the features, I expected more.",
    "Absolutely terrible! Never buying from this brand again.",
    "The packaging was damaged but the product is working fine.",
    "I'm super excited about this product, can't wait to try it!",
    "It's overpriced for the quality you're getting, very disappointed."
]}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Step 1: Sentiment Analysis using VADER
def analyze_sentiment_vader(text):
    """Returns the compound sentiment score for a given text."""
    score = sia.polarity_scores(text)['compound']
    return score

# Apply sentiment analysis to each row
df['sentiment_score'] = df['text'].apply(analyze_sentiment_vader)

# Classify sentiment based on score: positive (>0), negative (<0), neutral (0)
df['sentiment_label'] = df['sentiment_score'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

# Step 2: Visualize Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_label', data=df, palette='coolwarm')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Step 3: Word Cloud for Positive and Negative Sentiment
positive_text = ' '.join(df[df['sentiment_label'] == 'Positive']['text'])
negative_text = ' '.join(df[df['sentiment_label'] == 'Negative']['text'])

# Word Cloud for Positive Sentiment
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')
plt.axis('off')
plt.show()

# Word Cloud for Negative Sentiment
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')
plt.axis('off')
plt.show()

# Step 4: Sentiment Score Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['sentiment_score'], kde=True, bins=20, color='blue')
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Step 5: TextBlob Sentiment Analysis (Optional)
# You can also use TextBlob for sentiment polarity (between -1 and 1)
def analyze_sentiment_textblob(text):
    """Returns the sentiment polarity score using TextBlob."""
    return TextBlob(text).sentiment.polarity

df['textblob_score'] = df['text'].apply(analyze_sentiment_textblob)

# Compare VADER and TextBlob Scores
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sentiment_score', y='textblob_score', data=df, hue='sentiment_label', palette='coolwarm', s=100)
plt.title('Comparison of VADER vs TextBlob Sentiment Scores')
plt.xlabel('VADER Sentiment Score')
plt.ylabel('TextBlob Sentiment Score')
plt.show()
