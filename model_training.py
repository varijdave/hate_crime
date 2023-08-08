# model_training.py
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

def build_lstm_model(input_dim, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=output_dim))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_features, train_labels, test_features, test_labels, epochs=5, batch_size=64):
    model.fit(train_features, train_labels, validation_data=(test_features, test_labels), epochs=epochs, batch_size=batch_size)
    return model
