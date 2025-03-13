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

    :param text: input text
    :return: processed text (remove stopwords, lowercase all, remove extra whitespace)
    """

    text = text.lower()
    text = " ".join(text.split())  # remove extra whitespace

    with open('extra_stopwords.txt') as f:
        extra_stopwords = [l.strip() for l in f.readlines()]  # extra stopwords that show up often

    stop_words = set(stopwords.words('english'))
    stop_words.update(extra_stopwords)
    words = word_tokenize(text)
    words = ' '.join([word for word in words if word not in stop_words])

    return words

def chunk_text(text, tokenizer, chunk_size=200, overlap=50):
    """

    :param text: input text
    :param tokenizer: tokenizer to use
    :param chunk_size: chunk size to chunk into
    :param overlap: overlap for chunking
    :return: chunked text
    """
    tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks

def preprocess_and_chunk(text, tokenizer_name, chunk_size=200, overlap=50):
    """
    calls the preprocess & chunking functions on a text block and returns it
    splits text into sentences first to try to avoid going over the token limit for chunking
    """
    # preprocess
    cleaned_text = preprocess_text(text)

    # split the text into sentences to avoid going above the maximum sequence length as much as possible
    sentences = sent_tokenize(cleaned_text)

    # create tokenizer to use in chunk_text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # parallel processing for better runtime
    all_chunks = Parallel(n_jobs=4)(  # 4 cpu cores - my mac does not have CUDA cores
        delayed(chunk_text)(sentence, tokenizer, chunk_size, overlap)
        for sentence in sentences
    )

    # flatten list back into 1d
    return [chunk for sublist in all_chunks for chunk in sublist]

def preprocess_folder(directory, tokenizer_name, chunk_size=200, overlap=50):
    """
    run preprocessing_and_chunk on a folder of pdf files
    """
    out = []

    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            out += preprocess_and_chunk(text, tokenizer_name, chunk_size, overlap)

    out = [[''.join(ch for ch in o if (ch.isalnum()) or ch.isspace())] for o in out]
    return out

def main():
    # sample usage with preprocess_and_chunk
    text = 'This is a long document that needs to be split into smaller chunks. ' * 100
    chunks = preprocess_and_chunk(
        text,
        tokenizer_name='sentence-transformers/all-MiniLM-L6-v2',
        chunk_size=200,
        overlap=10  # arbitrary values
    )

    # Print the chunks
    for i, chunk in enumerate(chunks):
        print(f'Chunk {i+1}: {chunk}')

if __name__ == '__main__':
    main()