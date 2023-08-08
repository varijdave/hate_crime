import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Load the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/varijdave/hate_crime/main/labeled_data.csv')

# Preprocess text data
dataset['tweet'] = dataset['tweet'].str.lower()
dataset['tweet'] = dataset['tweet'].str.replace('[^\w\s]', '')  # Remove special characters

# Split the dataset into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    dataset['tweet'], dataset['class'], test_size=0.2, random_state=42
)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Convert sparse features to dense arrays
train_features_dense = train_features.toarray()
test_features_dense = test_features.toarray()

# Define and compile the model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=train_features.shape[1]))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_features_dense, train_labels, validation_data=(test_features_dense, test_labels), epochs=5, batch_size=100)
