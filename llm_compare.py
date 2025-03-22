"""
file to compare different factors (database type, chunk size, overlap, tokenizer, model type)
records time, response, and memory taken
"""

import time
import psutil  # tracks memory usage
import os
import pandas as pd
from db_functions import insert_query

# function to measure memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Return memory usage in MB

# actual logic
def run_tests(db_types, chunk_sizes, overlaps, tokenizer_names, models, question):
    results = []

    for db_type in db_types:
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                for tokenizer_name in tokenizer_names:
                    for model in models:
                        print(f"Testing: db_type={db_type}, chunk_size={chunk_size}, overlap={overlap}, tokenizer_name={tokenizer_name}, model={model}")

                        # Record start time and memory
                        start_time = time.time()
                        start_memory = get_memory_usage()

                        # Run the insert_query function
                        try:
                            response = insert_query(db_type, chunk_size, overlap, tokenizer_name, model, question)
                        except Exception as e:
                            response = f"Error: {str(e)}"

                        # Record end time and memory
                        end_time = time.time()
                        end_memory = get_memory_usage()

                        # Calculate time and memory usage
                        time_taken = end_time - start_time
                        memory_used = end_memory - start_memory

                        # Store results
                        results.append({
                            'db_type': db_type,
                            'chunk_size': chunk_size,
                            'overlap': overlap,
                            'tokenizer_name': tokenizer_name,
                            'model': model,
                            'time_taken': time_taken,
                            'memory_used': memory_used,
                            'response': response
                        })

                        # Print results for this combination
                        print(f"Time taken: {time_taken:.2f} seconds")
                        print(f"Memory used: {memory_used:.2f} MB")
                        print(f"Response: {response}")
                        print("-" * 50)

    return results

# Function to save results to a CSV file
def save_results_to_csv(results, filename="test_results.csv"):
    # insert list of dicts, outputs file
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Run the tests and save results
if __name__ == '__main__':
    db_types = ['faiss', 'chroma']
    chunk_sizes = [100, 200, 500]
    overlaps = [10, 20, 50]
    tokenizer_names = [
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    models = ['llama2', 'mistral']
    question = "What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?"

    results = run_tests(db_types, chunk_sizes, overlaps, tokenizer_names, models, question)
    save_results_to_csv(results)