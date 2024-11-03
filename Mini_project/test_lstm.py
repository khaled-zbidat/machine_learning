import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw  # Use Pillow for image manipulation
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import os

# Load the saved model
model = tf.keras.models.load_model('word_width_model.h5')
with open("tokenizer.json") as file:
    tokenizer_json = json.load(file)
    tokenizer = tokenizer_from_json(tokenizer_json)

print("Model loaded from 'word_width_model.h5'")
print(f"Model input shape: {model.input_shape}")  # Check input shape

# Load the training data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess function to create char_to_index from the dataset
def create_char_to_index(df):
    return {char: idx for idx, char in enumerate(sorted(set(''.join(df['Word'].values))), start=1)}

# Main script to test the model
if __name__ == '__main__':
    file_path = 'store2.csv'

    # Specify the image file to draw bounding boxes on
    fixed_height = 30  # Adjust as needed
    white_space = 20
    image_path = "ilovepdf_pages-to-jpg/page_1.jpg"  # Replace with your image file
    
    # Load data to create char_to_index
    df = load_data(file_path)
    char_to_index = create_char_to_index(df)


    x_start = None
    y_line = None
    modified_image_path = "output_image.jpg"  # Change the output filename as needed
    with Image.open(image_path) as img:
        img.save(modified_image_path)

    while True:
        words = input("Enter a word or sentence (or 'exit' to quit): ")
        if words.lower() == 'exit':
            break
        
        for word in words.split(" "):
            # Prepare input for prediction by tokenizing the word
            word_sequence = tokenizer.texts_to_sequences([word])[0]  # Get sequence for the word
            word_input = np.array(word_sequence).reshape(-1, 1)  # Reshape to (sequence_length, 1)

            # Predict the width using the model
            try:
                if x_start is None:
                    x_start = float(input("Enter the x-coordinate for the start of the line: "))

                if y_line is None:
                    y_line = float(input("Enter the y-coordinate for the line: "))

                predicted_width = model.predict(word_input)
                average_width = np.mean(predicted_width)  # Average width prediction
                print(f'Predicted width for "{word}": {average_width}')

                # Input coordinates for the line and where to place the bounding box
                
                # Calculate the bounding box coordinates
                bounding_box = [
                    x_start, 
                    y_line, 
                    x_start + average_width, 
                    y_line + fixed_height
                ]  # Use fixed height for bounding box
                x_start += average_width + white_space

                fault_amount = 0
                # Open the image and draw the bounding box
                with Image.open(modified_image_path) as img:
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(bounding_box, outline="green", width=2)  # Draw a green rectangle

                    # Show the image with the bounding box
                    img.show()
                    # Confirm with user if the bounding box is correct
                    feedback = input("Is the bounding box correct? (yes/no): ").strip().lower()
                    if feedback == 'no':
                        fault_direction = input("Indicate the direction of fault (right/left): ")
                        fault_amount = float(input("How much was the fault (in units): "))
                        # Adjust bounding box based on feedback (this part can be expanded as needed)
                        if fault_direction == 'left':
                            bounding_box[0] -= fault_amount
                        else:
                            bounding_box[2] += fault_amount
                            x_start += fault_amount

                    else:# Save the modified image with bounding boxes
                        img.save(modified_image_path)
                        print(f"Bounding boxes have been added to modified image.")
                if fault_amount:
                    with Image.open(modified_image_path) as img:
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(bounding_box, outline="red", width=5)  # Draw a green rectangle
                        img.save(modified_image_path)
                        print(f"Bounding boxes have been added to modified image.")

            except Exception as e:
                print(f"Error during prediction: {e}")

    # Open the last modified image automatically
    # os.system(f'xdg-open "{modified_image_path}"')









'''
import tensorflow as tf
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import os

# Load the saved model
model = tf.keras.models.load_model('word_width_model.h5')
print("Model loaded from 'word_width_model.h5'")
print(f"Model input shape: {model.input_shape}")  # Check input shape

# Load the training data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess function to create char_to_index from the dataset
def create_char_to_index(df):
    return {char: idx for idx, char in enumerate(sorted(set(''.join(df['Word'].values))), start=1)}

# Main script to test the model
if __name__ == '__main__':
    file_path = 'store.csv'
    
    # Load data to create char_to_index
    df = load_data(file_path)
    char_to_index = create_char_to_index(df)

    # Specify the PDF file to draw bounding boxes on
    pdf_path = 'book1.pdf'
    pdf_document = fitz.open(pdf_path)

    # Set a fixed height for the bounding box
    fixed_height = 20  # Adjust as needed

    while True:
        word = input("Enter a word (or 'exit' to quit): ")
        if word.lower() == 'exit':
            break
        
        # Prepare input for prediction
        word_sequence = np.array([char_to_index.get(char, 0) for char in word], dtype=np.float32)
        
        # Reshape input to match the model's expected input shape
        # Assuming the model expects (None, 1), we reshape to (len(word), 1)
        word_input = word_sequence.reshape(-1, 1)  # Reshape to (sequence_length, 1)

        # Predict the width using the model
        try:
            predicted_width = model.predict(word_input)
            average_width = np.mean(predicted_width)  # Average width prediction
            print(f'Predicted width for "{word}": {average_width}')

            # Input coordinates for the line and where to place the bounding box
            x_start = float(input("Enter the x-coordinate for the start of the line: "))
            y_line = float(input("Enter the y-coordinate for the line: "))
            
            # Calculate the bounding box coordinates
            bounding_box = [
                x_start, 
                y_line - fixed_height / 2, 
                x_start + average_width, 
                y_line + fixed_height / 2
            ]  # Use fixed height for bounding box
            
            # Draw the bounding box on the PDF
            page = pdf_document[0]  # Assuming you're working with the first page
            page.draw_rect(bounding_box, color=(0, 1, 0), width=2)  # Draw a green rectangle

            # Confirm with user if the bounding box is correct
            feedback = input("Is the bounding box correct? (yes/no): ").strip().lower()
            if feedback == 'no':
                fault_direction = input("Indicate the direction of fault (right/left): ")
                fault_amount = float(input("How much was the fault (in units): "))
                # Adjust bounding box based on feedback (this part can be expanded as needed)

        except Exception as e:
            print(f"Error during prediction: {e}")

    # Save the modified PDF
    modified_pdf_path = 'modified_book1.pdf'
    pdf_document.save(modified_pdf_path)
    pdf_document.close()
    print(f"Bounding boxes have been added to '{modified_pdf_path}'.")

    # Open the modified PDF automatically
    os.system(f'xdg-open "{modified_pdf_path}"')
'''
