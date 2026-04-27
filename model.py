import os
import time
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class NextWordPredictor:
    def __init__(self, data_path='cleaned_data.txt', model_path='next_word_model.keras', tokenizer_path='tokenizer.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer()
        self.model = None
        self.max_sequence_len = 0
        self.vocab_size = 0

    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split text into lines
        lines = text.split('\n')
        
        print("Fitting tokenizer...")
        self.tokenizer.fit_on_texts(lines)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print(f"Vocabulary size: {self.vocab_size}")

        print("Generating input sequences...")
        input_sequences = []
        for line in lines:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            # Create n-gram sequences to predict the next word
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
                
        # To avoid massive padding that hurts memory and training time, we cap sequence length.
        # E.g. we only need the last ~20 words to predict the next one.
        max_actual_len = max([len(seq) for seq in input_sequences] if input_sequences else [0])
        self.max_sequence_len = min(max_actual_len, 20)
        
        print(f"Max sequence length (capped for efficiency): {self.max_sequence_len}")
        
        print("Padding sequences...")
        # pad_sequences by default truncates from the beginning ('pre'), keeping the most recent words.
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')
        
        # Split into predictors (X) and label (y)
        X = input_sequences[:, :-1]
        y = input_sequences[:, -1]
        
        return X, y

    def build_model(self):
        print("Building model architecture...")
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, 300, input_length=self.max_sequence_len - 1))
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(LSTM(256))
        self.model.add(Dense(512, activation='relu'))
        # We use sparse_categorical_crossentropy to avoid one-hot encoding the huge vocabulary
        self.model.add(Dense(self.vocab_size, activation='softmax'))
        
        # Using Adam with a slightly custom learning rate can sometimes yield better convergence
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()

    def train(self, X, y, epochs=10, batch_size=128):
        print("Starting training...")
        
        # Stop training if the accuracy stops improving for 15 epochs
        early_stopping = EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True, verbose=1)
        # Reduce learning rate when the accuracy plateaus to fine-tune weights
        reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        
        self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )
        
        # Save the trained model and tokenizer
        self.model.save(self.model_path)
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump({'tokenizer': self.tokenizer, 'max_sequence_len': self.max_sequence_len}, handle)
        print(f"Model saved to {self.model_path}")
        print(f"Tokenizer saved to {self.tokenizer_path}")

    def load_existing_model(self):
        print(f"Loading existing model from {self.model_path}...")
        self.model = load_model(self.model_path)
        with open(self.tokenizer_path, 'rb') as handle:
            data = pickle.load(handle)
            self.tokenizer = data['tokenizer']
            self.max_sequence_len = data['max_sequence_len']
        print("Model and tokenizer loaded successfully.")

    def predict_next_word(self, text):
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded.")
            
        # Convert input text to tokens
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        # Pad to match model input shape
        token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
        
        # Predict the probabilities for the next word
        predicted_probs = self.model.predict(token_list, verbose=0)
        predicted_class = np.argmax(predicted_probs, axis=-1)[0]
        
        # Find the word corresponding to the predicted class index
        output_word = ""
        for word, index in self.tokenizer.word_index.items():
            if index == predicted_class:
                output_word = word
        return output_word

    def predict_next_n_words(self, text, n=10):
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded.")
            
        current_text = text
        predicted_words = []
        
        for _ in range(n):
            next_word = self.predict_next_word(current_text)
            if not next_word:
                break
            predicted_words.append(next_word)
            current_text += " " + next_word
            
        return predicted_words

if __name__ == "__main__":
    predictor = NextWordPredictor()
    
    print("Preparing data for training...")
    X, y = predictor.load_data()
    
    predictor.build_model()
    # Decreasing batch size allows the model to learn more frequently per epoch.
    # Increasing epochs is safe now because EarlyStopping will halt it automatically.
    predictor.train(X, y, epochs=150, batch_size=64)
        
    print("Training/Loading Complete.")
    print("To interactively predict words, please run: python predict.py")