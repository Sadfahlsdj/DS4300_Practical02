import time
import tracemalloc
from preprocess import preprocess_folder
import pandas as pd

def measure_performance(directory, tokenizer_name, chunk_size, overlap):
    """
    Measure memory usage and speed of the preprocessing function.
    """
    # Start tracking memory usage
    tracemalloc.start()

    # Start timing
    start_time = time.time()

    # Run the preprocessing function
    chunks = preprocess_folder(directory, tokenizer_name, chunk_size, overlap)

    # Stop timing
    end_time = time.time()

    # Stop tracking memory usage
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate elapsed time
    elapsed_time = end_time - start_time

    return {
        "tokenizer": tokenizer_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "time_seconds": elapsed_time,
        "memory_usage_mb": peak_memory / (1024 * 1024),  # Convert to MB
    }

def main():
    tokenizers = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "roberta-base",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    chunk_sizes = [128, 256, 500]
    overlap_sizes = [0, 50, 100] # overlap needs to be smaller than chunk
    directory = 'pdf_files'

    # Store results
    results = []

    for tokenizer_name in tokenizers:
        for chunk_size in chunk_sizes:
            for overlap in overlap_sizes:
                print(f"Testing: {tokenizer_name}, chunk_size={chunk_size}, overlap={overlap}")
                result = measure_performance(directory, tokenizer_name, chunk_size, overlap)
                results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('results.csv')

if __name__ == '__main__':
    main()