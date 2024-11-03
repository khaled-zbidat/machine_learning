import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import json

# Load the CSV data
def load_data():
    # Load words CSV (Word, Width)
    df = pd.read_csv('store2.csv')  # Ensure this matches your file name and path
    print("DataFrame loaded successfully:")
    print(df.head())  # Print the first few rows to check
    print("Columns in DataFrame:", df.columns.tolist())  # Check columns
    return df

tokenizer = Tokenizer()

def prep_tokenizer(words):    
    tokenizer.fit_on_texts(words)
    vocab_size = len(tokenizer.word_index) + 1
    return vocab_size

# Build the model
def build_model(vocab_size, embedding_dim=64):

    # Input for Word (encoded as integer)
    word_input = layers.Input(shape=(1,), dtype=tf.int32)  # Word index input
    word_embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(word_input)
    word_embedding = layers.Flatten()(word_embedding)  # Flatten embedding output
    
    # Additional Dense Layers for Processing
    word_features = layers.Dense(256, activation='relu')(word_embedding)
    word_features = layers.Dense(64, activation='relu')(word_embedding)
    output = layers.Dense(1, activation='linear')(word_features)  # Predict width

    # Model: Word -> Embedding -> Width Prediction
    model = models.Model(inputs=[word_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Training loop
def train_model(model, df, epochs=10, batch_size=32):
    for epoch in range(epochs):
        # Shuffle data
        df = df.sample(frac=1).reset_index(drop=True)

        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]

            # Here we assume 'Width' is already in the correct format (float)
            batch_widths = batch['Width'].values.astype(float)  # Ensure it's float
            
            # Prepare input (word widths)
            batch_inputs = batch['Word'].values # Using Words as input
            batch_sequences = tokenizer.texts_to_sequences(batch_inputs)  # Convert words to sequences

            # Flatten each list in batch_sequences to a single integer (assuming each word is tokenized to one token)
            batch_inputs = np.array([seq[0] if seq else 0 for seq in batch_sequences]).reshape(-1, 1)

            
            # Train the model on this batch
            loss = model.train_on_batch(batch_inputs, batch_widths)
            print(f'Epoch {epoch + 1}/{epochs}, Batch {i // batch_size + 1}, Loss: {loss:.4f}')


# Main script to run training
if __name__ == '__main__':
    # Load data
    df = load_data()
    
    vocab_size = prep_tokenizer(df['Word'].tolist())
    # Build and compile the model
    model = build_model(vocab_size)
    
    # Train the model
    train_model(model, df, epochs=30, batch_size=64)
    
    # Save the model for later use
    model.save('word_width_model.h5')  # Save the weights of the model
    with open("tokenizer.json", "w") as file:
        json.dump(tokenizer.to_json(), file)
    print("Model saved to 'word_width_model.h5'")

