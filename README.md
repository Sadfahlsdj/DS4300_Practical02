# DS4300 Practical 2

The preprocess.py file contains the functions that are used to preprocess and chunk text. 
This file does not need to be run, but contains sample usage in its main function.

The db_functions.py file contains functions used to insert & query documents into a Chroma or Redis database (more databases coming soon).
It does not need to be run, but contains sample usage in its main function.

To run with a redis database, a local docker container with the redis:latest image should be named ds4300_practical02 and run on port 6379.
Unfortunately, due to hardware restrictions, the fact that Redis is very obnoxious to run without Docker makes it computationally infeasible to use.
Users with laptops with similar compute to mine (M1 chip, 8GB total RAM) may run into the same issue. Other databases are run without docker.

The llm_compare.py file contains functions to test different parameters (database type, chunk size, overlap, tokenizer type, and model).
When run, it will record the time taken, memory usage, and output to a preset question per combination of parameters.
A runtime of a couple hours can be expected on a device with specs similar to mine.
Due to hardware limitations, the memory usage tracking is unfortunately very bugged, with negative memory usages being recorded alarmingly often.
Thus, I only considered time taken and my own judgement of response quality in choosing the parameters to use.
The outputs from this function can be found in llm_test_results.csv

I chose the faiss database, with a chunk size of 200 and overlap of 10, using the sentence-transformers/multi-qa-MiniLM-L6-cos-v1 transformer
and llama2 model, as my preferred parameter combination.

To run the actual LLM, run the query_rag.py file. It does not take any command line arguments.
It will prompt the user for a question, print the LLM response after some time, and repeat. The user can quit at any time.
Once the user quits, it will log all questions and responses. The program will create a logs folder the first time it will run to store these.

The program will repeat every LLM prompt that is inputted. If the prompt does not match what a user typed, they should force quit the program, wait a little, and restart it and try again. This phenomenon is likely a symptom of the program grabbing junk data due to running out of memory, and is hard to avoid on devices with lower compute.
