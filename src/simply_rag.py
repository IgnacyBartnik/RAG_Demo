import ollama
import numpy as np
import tqdm
from sklearn.metrics.pairwise import cosine_similarity



def add_chunk_to_database(chunk, VECTOR_DB, EMBEDDING_MODEL):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

def cosine_sim_sklearn_vecs(a, b):
    """
    Calculate cosine similarity between two vectors using sklearn.
    """
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0, 0]

def retrieve(query, EMBEDDING_MODEL, VECTOR_DB, top_n=3):
    # Calculate the embedding for the query
    query_emb = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    # Calculate cosine similarity for each chunk in the vector database
    similarities = [(chunk, cosine_sim_sklearn_vecs(query_emb, emb)) for chunk, emb in VECTOR_DB]
    # Sort the similarities in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]



def main():
    EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
    LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

    dataset = []
    with open('cat-facts.txt', 'r') as file:
        dataset = file.readlines()
    print(f'Loaded {len(dataset)} entries')


    VECTOR_DB = []
    # Let us check if the vectorization has already been done
    try:
        with open('vector_db.txt', 'r') as file:
            for line in file:
                chunk, embedding_str = line.strip().split('\t')
                embedding = list(map(float, embedding_str.split(',')))
                VECTOR_DB.append((chunk, embedding))
        print(f'Loaded {len(VECTOR_DB)} entries from vector_db.txt')
        vectorized_dataset_loaded = True
    except FileNotFoundError:
        print('vector_db.txt not found, proceeding to vectorize the dataset')
        for i, chunk in tqdm(enumerate(dataset), total=len(dataset), desc="Adding chunks"):
            add_chunk_to_database(chunk, VECTOR_DB, EMBEDDING_MODEL)
        vectorized_dataset_loaded = False
        

    # Save the vector database to a txt file
    if not vectorized_dataset_loaded:
        print('Saving the vector database to vector_db.txt')
        with open('vector_db.txt', 'w') as file:
            for chunk, embedding in VECTOR_DB:
                file.write(f"{chunk.strip()}\t{','.join(map(str, embedding))}\n")


    # Chatbot

    input_query = input('Ask me a question: ')
    retrieved_knowledge = retrieve(input_query, EMBEDDING_MODEL, VECTOR_DB)

    print('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = (
        "You are a helpful chatbot.\n"
        "Use only the following pieces of context to answer the question. "
        "Don't make up any new information:\n"
        + "\n".join([f' - {chunk}' for chunk, _ in retrieved_knowledge])
    )
    # print(instruction_prompt)

    stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': input_query},
    ],
    stream=True,
    )

    # print the response from the chatbot in real-time
    # print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)



if __name__ == "__main__":
    main()