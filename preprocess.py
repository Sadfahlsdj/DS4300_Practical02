import os
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers import AutoTokenizer, AutoModel
import torch

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

    :param text: input text
    :return: processed text (remove stopwords, lowercase all, remove extra whitespace)
    """

    text = text.lower()
    text = " ".join(text.split())  # remove extra whitespace
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text) # remove non alphanumerics

    with open('extra_stopwords.txt') as f:
        extra_stopwords = [l.strip() for l in f.readlines()]  # extra stopwords that show up often

    stop_words = set(stopwords.words('english'))
    stop_words.update(extra_stopwords)
    words = word_tokenize(text)
    words = ' '.join([word for word in words if word not in stop_words])

    return words

def chunk_text(text, tokenizer, chunk_size=200, overlap=50):
    """
    Chunk the text into smaller pieces - sentence splitting is done here
    :param text: input text
    :param tokenizer: tokenizer to use
    :param chunk_size: chunk size for chunking
    :param overlap: overlap for chunking
    :return: input text broken into chunks
    """
    # sentence splitting
    sentences = sent_tokenize(text)

    # tokenize sentences first
    tokens = []
    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, truncation=False, add_special_tokens=False)
        tokens.extend(sentence_tokens)

    # chunk sentences
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))

    return chunks

def preprocess_chunk_embed(text, tokenizer_name, chunk_size=200, overlap=50):
    """
    Calls the preprocess & chunking functions on a text block
    Returns the original & cleaned chunks, and embeddings of cleaned chunks
    Splits text into sentences first to try to avoid going over the token limit for chunking.
    Returns:
        - original_chunks: List of original text chunks to be returned later by queries
        - cleaned_chunks: List of preprocessed text chunks
        - embeddings: List of corresponding embeddings of cleaned chunks for AI stuff
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    model = AutoModel.from_pretrained(tokenizer_name)

    original_chunks = chunk_text(text, tokenizer, chunk_size, overlap)

    # preprocess each chunk individually
    # needed to ensure same # of chunks for original & preprocessed text
    cleaned_chunks = []
    for chunk in original_chunks:
        cleaned_chunk = preprocess_text(chunk)
        cleaned_chunks.append(cleaned_chunk)

    # tokenize chunks for embedding
    inputs = tokenizer(cleaned_chunks, padding=True, truncation=True, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    # chroma needs embeddings as a numpy array
    embeddings = embeddings.numpy()

    # make sure chunk len matches
    if len(original_chunks) != len(cleaned_chunks):
        raise ValueError(f'Length mismatch: original_chunks ({len(original_chunks)}) != cleaned_chunks ({len(cleaned_chunks)})')

    return original_chunks, cleaned_chunks, embeddings

def process_folder(directory, tokenizer_name, chunk_size=200, overlap=50):
    """
    Run preprocessing_chunk_embed on a folder of PDF files
    """
    original, clean, embeddings_all, metadatas = [], [], [], []

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            raw_chunks, clean_chunks, embeddings = (
                preprocess_chunk_embed(text, tokenizer_name, chunk_size, overlap))
            original.extend(raw_chunks)
            clean.extend(clean_chunks)
            embeddings_all.extend(embeddings)  # Add embeddings to the output list

            for chunk in raw_chunks:
                metadatas.append({
                    'source': filename,  # Store the source file name
                    'original_text': chunk  # Store the original text for display
                })

    return original, clean, embeddings_all, metadatas

def main():
    # sample usage with preprocess_folder (pdf_test folder has a single document)
    a = process_folder('pdf_test', 'sentence-transformers/all-MiniLM-L6-v2',
                       200, 10)
    print(a)

if __name__ == '__main__':
    main()