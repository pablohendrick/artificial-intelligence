import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # Download NLTK resources if necessary

# Load the dataset directly into the code (replace with your own data)
data = {
    'inappropriate_words': [
        'word1', 'word2', 'word3', 'hate1', 'hate2'
    ]
}

# Convert data to a DataFrame (if it's in a list or dictionary format)
df_inappropriate_words = pd.DataFrame(data)

# Function to load the text to be cleaned
def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print("File not found. Check the file path and name.")
        return None

# Advanced text preprocessing function
def advanced_text_preprocessing(text):
    text = text.lower()  # Convert all letters to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize using NLTK
    return tokens

# Function to clean the text by removing inappropriate words
def clean_text_remove_inappropriate(text, clean_tokens):
    text_tokens = word_tokenize(text.lower())  # Tokenize the provided text
    clean_tokens = set(clean_tokens)  # Convert the list of clean tokens to a set for efficient lookup

    clean_text = ' '.join([token for token in text_tokens if token not in clean_tokens])
    return clean_text

# Example usage:
file_path_text = 'path/to/your/file.txt'
text_to_clean = load_text(file_path_text)

if text_to_clean:
    clean_tokens = advanced_text_preprocessing(text_to_clean)
    cleaned_text = clean_text_remove_inappropriate(text_to_clean, clean_tokens)
    print("Cleaned text removing inappropriate words:")
    print(cleaned_text)
