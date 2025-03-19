import chromadb
from chromadb.config import Settings
from preprocess import extract_text_from_pdf, process_folder
import os
import ollama

# Step 1: Connect to the ChromaDB server running in Docker
def load_collection(collection_name):
    client = chromadb.Client(Settings(
        chroma_server_host='localhost',  # need this
        chroma_server_http_port='8000'   # default Chroma port is 8000, but I set this to 8000 too
    ))

    collection = client.get_or_create_collection(name=collection_name)
    return collection

def insert_documents(collection, ids, documents, embeddings, metadatas):
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


def insert_folder(collection, folder, tokenizer_name, chunk_size, overlap):
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
    insert_documents(collection, ids, chunked_docs, embeddings, metadatas)

# Step 4: Query the collection
def query_db(query, n_results, collection):
    """

    :param query: list of queries
    :param n_results: # of results to show
    :param collection: link to the database collection
    :return: results of query, only includes metadatas (raw text) for readability
    """
    all_documents = collection.get(
        include=['metadatas', 'documents']
    )

    # get all documents
    context = " ".join(all_documents['documents'])
    documents = all_documents['documents']

    prompt = (f"Use the following context to answer the question. "
              f"Context: {context}. Question: {query[0]}. Answer:")

    response = ollama.generate(
        model="llama2",  # model goes here - mistral, llama2, etc
        prompt=prompt
    )

    # query_results = collection.query(
    #     query_texts=query,
    #     n_results=n_results,
    #     include=['metadatas'] # metadata has the raw text so it's the most useful to us
    # )
    #
    # return query_results
    return response['response']

def main():
    # example usage
    # untested on a folder with >1 document inside but I am going crazy doing this already
    coll = load_collection('ds4300_p2')
    insert_folder(coll, 'pdf_files', 'sentence-transformers/all-MiniLM-L6-v2',
                          200, 10)

    response = query_db(['What are the benefits of a relational model'], 1, coll)
    print(response)

if __name__ == '__main__':
    main()