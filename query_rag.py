from db_functions import insert_folder, query_db, load_collection
import time
import os

def main():
    chunk_size, overlap, db = 200, 10, 'faiss'
    transformer, model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 'llama2'
    # values i chose

    filename = f'{time.time()}.txt' # logging stuff
    try:
        os.mkdir('logs') # create logging folder
        print(f"Directory 'logs' created successfully.")
    except:
        pass

    # initialize db
    coll = load_collection('ds4300_practical02', db, transformer)
    insert_folder(coll, 'pdf_files', transformer, chunk_size, overlap, db)

    while True:
        question = input('Input question to ask the LLM; insert a Q (case insensitive) to quit\n'
                         'Allow the LLM around a minute or two to think\n')
        if question.lower() == 'q':
            break

        r_raw = query_db([question], coll, db, model)
        response = r_raw + '\n------------------------------------\n'
        print(response)
        with open(f'logs/{filename}', 'a') as f:
            log_response = (f'Question: {question}\n------------------------------------\n'
                            f'Response: {response}')
            f.write(log_response)

if __name__ == '__main__':
    main()