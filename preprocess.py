import os
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
from transformers import AutoTokenizer
from joblib import Parallel, delayed

def extract_text_from_pdf(pdf_path):
    """

    :param pdf_path: path to singular pdf file
    :return: raw text from that file
    """
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text
def preprocess_text(text):
    """

    """

    text = text.lower()
    text = " ".join(text.split())  # Remove extra whitespace

    with open('extra_stopwords.txt') as f:
        extra_stopwords = [l.strip() for l in f.readlines()]  # extra stopwords that show up often

    stop_words = set(stopwords.words('english'))
    stop_words.update(extra_stopwords)
    words = word_tokenize(text)
    words = ' '.join([word for word in words if word not in stop_words])

    return words

def chunk_text(text, tokenizer, chunk_size=200, overlap=50):
    """
    Optimized chunking function.
    """
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

def preprocess_and_chunk(text, tokenizer_name, chunk_size=200, overlap=50):
    """
    Preprocess text, split it into sentences, and chunk each sentence.
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text)

    # Split the text into sentences to avoid going above the maximum sequence length as much as possible
    sentences = sent_tokenize(cleaned_text)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Process sentences in parallel
    all_chunks = Parallel(n_jobs=4)(  # Use 4 CPU cores
        delayed(chunk_text)(sentence, tokenizer, chunk_size, overlap)
        for sentence in sentences
    )

    # Flatten the list of chunks
    return [chunk for sublist in all_chunks for chunk in sublist]

def preprocess_folder(directory, tokenizer_name, chunk_size=200, overlap=50):
    out = []

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            out += preprocess_and_chunk(text, tokenizer_name, chunk_size, overlap)

    out = [[''.join(ch for ch in o if (ch.isalnum()) or ch.isspace())] for o in out]
    return out

def main():
    # Example usage
    text = "This is a long document that needs to be split into smaller chunks. " * 100
    chunks = preprocess_and_chunk(
        text,
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=200,
        overlap=10  # Reduced overlap for faster processing
    )

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")

if __name__ == '__main__':
    main()