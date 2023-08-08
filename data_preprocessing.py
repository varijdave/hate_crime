import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/varijdave/hate_crime/main/labeled_data.csv')

# Preprocess text data
dataset['text'] = dataset['text'].str.lower()
dataset['text'] = dataset['text'].str.replace('[^\w\s]', '')  # Remove special characters

# Split the dataset into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['text'], dataset['label'], test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)
