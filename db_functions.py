import chromadb
from chromadb.config import Settings
from preprocess import extract_text_from_pdf, process_folder
import os
import ollama
import redis
import faiss # third vector db
import numpy as np
from transformers import AutoTokenizer, AutoModel

# connect to chroma server, returns the collection
def load_collection(collection_name, db_type, tokenizer):
    model = AutoModel.from_pretrained(tokenizer)
    embedding_size = model.config.hidden_size # get embedding size of tokenizer
    if db_type.lower() == 'chroma':
        client = chromadb.Client(Settings(
            chroma_server_host='localhost',  # need this
            chroma_server_http_port='8000'   # default Chroma port is 8000
        ))

        collection = client.get_or_create_collection(name=collection_name)
        return collection

    elif db_type.lower() == 'redis':
        client = redis.Redis(host='localhost', port=6379, db=0) # default redis port is 6379
        return client

    elif db_type.lower() == 'faiss':
        dimension = embedding_size  # adjust based on dimension of embedding
        index = faiss.IndexFlatL2(dimension)  # L2 metric for similarity search
        index.metadata_store = {} # used later to store documents & metadata
        return index

def insert_documents(collection, ids, documents, embeddings, metadatas, db_type):
    if db_type.lower() == 'chroma':
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    elif db_type.lower() == 'redis':
        for id, doc, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            collection.hset(f"doc:{id}", mapping={
                "document": doc,
                "embedding": embedding.tobytes(),  # convert np array to bytes
                "metadata": str(metadata)
            })

    elif db_type.lower() == 'faiss':
        # add numpy'd embeddings to collection
        embeddings_array = np.array(embeddings, dtype='float32')
        collection.add(embeddings_array)  # faiss only directly stores embeddings

        # store documents & metadata manually
        for id, doc, metadata in zip(ids, documents, metadatas):
            collection.metadata_store[id] = {'document': doc, 'metadata': metadata}

def insert_folder(collection, folder, tokenizer_name, chunk_size, overlap, db_type):
    """
    insert the embeddings, chunks, and raw docs (metadata) from a whole folder of documents;
    see main method for example usage
    :param collection: collection from load_collection
    :param folder: relative filepath to folder
    :param tokenizer_name: for use with preprocess_folder
    :param chunk_size: for use with preprocess_folder
    :param overlap: for use with preprocess_folder
    :return: nothing, just inserts documents
    """
    raw_docs, chunked_docs, embeddings, metadatas = (
        process_folder(folder, tokenizer_name, chunk_size, overlap))
    ids = [f"doc{i}" for i in range(len(chunked_docs))]
    insert_documents(collection, ids, chunked_docs, embeddings, metadatas, db_type)

# Step 4: Query the collection
def query_db(query, collection, db_type, model):
    """

    :param query: list of queries
    :param n_results: # of results to show
    :param collection: link to the database collection
    :return: results of query, only includes metadatas (raw text) for readability
    """
    if db_type.lower() == 'chroma':
        all_documents = collection.get(
            include=['metadatas', 'documents']
        )

        context = "\n----------------\n".join(all_documents['documents'])  # dashes help to separate files

    elif db_type.lower() == 'redis':
        all_documents = []
        for key in collection.scan_iter("doc:*"):
            doc_data = collection.hgetall(key)
            all_documents.append({
                'document': doc_data[b'document'].decode('utf-8'),
                'metadata': eval(doc_data[b'metadata'])
            })

        context = "\n----------------\n".join([d['document'] for d in all_documents])

    elif db_type.lower() == 'faiss':
        context = []
        for doc_id, data in collection.metadata_store.items(): # we created the metadata store earlier
            context.append(data['document'])
        context = "\n----------------\n".join(context)

    # get all documents
    prompt = (f"Use the following context to answer the question. "
              f"Context: {context}. Question: {query[0]}. Answer:")

    response = ollama.generate(
        model=model,  # model goes here - mistral, llama2, etc
        prompt=prompt
    )

    return response['response']

def insert_query(db_type, chunk_size, overlap, tokenizer_name, model, question):
    coll = load_collection('ds4300_practical02', db_type, tokenizer_name)
    insert_folder(coll, 'pdf_files', tokenizer_name,
                  chunk_size, overlap, db_type)

    response = query_db([question], coll, db_type, model)  # question needs to be inside a list
    return response


def main():
    # example usage
    # db_type = 'faiss' # can be chroma, redis, faiss (caps insensitive)
    # # I have confirmed that redis works, but since it needs docker it is very computationally taxing
    # # this means that it is unfortunately not feasible to use normally
    #
    # # collection name needs to be exact for redis I think
    # coll = load_collection('ds4300_practical02', db_type)
    # insert_folder(coll, 'pdf_files', 'sentence-transformers/all-MiniLM-L6-v2',
    #                       200, 10, db_type)
    #
    # questions = [ # from march20 inclass, decent set of testing questions
    #     'What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?',
    #     'When are linked lists faster than contiguously-allocated lists?',
    #     'Given an AVL tree with the numbers 30, 25, 35, and 20, what imbalance case is caused by inserting 23?',
    #     'Why is a B+ Tree a better than an AVL tree when indexing a large dataset?',
    #     'What is disk-based indexing and why is it important for database systems?',
    #     'In the context of a relational database system, what is a transaction?',
    #     'Succinctly describe the four components of ACID compliant transactions.',
    #     'Why does the CAP principle not make sense when applied to a single-node MongoDB instance?',
    #     'Describe the differences between horizontal and vertical scaling.',
    #     'Briefly describe how a key/value store can be used as a feature store.',
    #     'When was Redis originally released?',
    #     'In Redis, what is the difference between the INC and INCR commands?',
    #     'What are the benefits of BSON over JSON in MongoDB?',
    #     'Write a Mongo query based on the movies data set that returns the titles of all movies released between 2010 and 2015 from the suspense genre?',
    #     'What does the $nin operator mean in a Mongo query?'
    # ]
    #
    # model = 'llama2' # llama2, mistral7, llama3.1 etc
    # responses = []
    # for i, question in enumerate(questions[:1]): # remove index to test all
    #     response = query_db([question], coll, db_type, model) # question needs to be inside a list
    #     print(f'{i}: {response}')
    #     responses.append(response)
    #
    # with open('mar20_inclass_answers.txt', 'w') as f:
    #     f.writelines(responses)

    r = insert_query('chroma', 200, 10,
                 'sentence-transformers/all-MiniLM-L6-v2', 'mistral',
                 'What are the benefits of BSON over JSON in MongoDB?')
    print(r)

if __name__ == '__main__':
    main()