
import openai
import numpy as np
import faiss

global_model = "text-embedding-nomic-embed-text-v1.5"

# Example texts
documents = [
    "The Apollo program landed the first humans on the Moon.",
    "Artificial intelligence is transforming the world.",
    "NASA's Artemis program aims to return humans to the Moon.",
    "The Great Barrier Reef is facing significant environmental challenges.",
    "Quantum computing holds the potential for groundbreaking discoveries.",
    "Renewable energy sources are becoming increasingly important.",
    "The Amazon rainforest plays a crucial role in global climate regulation.",
    "Blockchain technology is impacting various industries.",
    "CRISPR gene editing offers new possibilities in medicine.",
    "The study of exoplanets is revealing the diversity of planetary systems.",
    "Sustainable agriculture practices are essential for food security.",
    "Virtual reality is creating immersive digital experiences.",
    "The human genome project mapped the entire human DNA sequence.",
    "Ocean currents have a major influence on global weather patterns.",
    "Nanotechnology is enabling advancements in materials science.",
    "The theory of relativity revolutionized our understanding of space and time.",
    "Conservation efforts are vital for protecting endangered species.",
    "Machine learning algorithms are improving data analysis techniques.",
    "The exploration of Mars continues to yield valuable scientific data.",
    "Urbanization presents both opportunities and challenges for cities."
]

DEBUG = 0

def dprint(s):
    if DEBUG == 1:
        print(s)


def generate_embedding(text_to_AI):
    LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed
    
    # Set up the OpenAI-compatible client
    client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')

    response = client.embeddings.create(
        model=global_model,
        input=text_to_AI
    )
    dprint(response)
    return response.data[0].embedding
    #return response["data"][0]["embedding"]


def main():
    # Generate embeddings
    embeddings = [generate_embedding(doc) for doc in documents]
    #embeddings = [ollama.embeddings(model="nomic-embed-text", prompt=doc)['embedding'] for doc in documents]
    dprint(embeddings)

    # Convert to NumPy array for FAISS
    embeddings_np = np.array(embeddings).astype('float32')
    dprint(embeddings_np)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance
    index.add(embeddings_np)  # Add vectors to index

    # Query Example
    query = "Tell me about the Moon landing"
    query_embedding = np.array(generate_embedding(query)).astype('float32').reshape(1,-1)
    #query_embedding = np.array([ollama.embeddings(model="nomic-embed-text", prompt=query)['embedding']]).astype('float32')
    dprint(query_embedding)

    # Search for similar documents
    D, I = index.search(query_embedding, k=5)  # Retrieve top 2 closest matches
    dprint("D:\n")
    dprint(D)
    dprint("I:\n")
    dprint(I)

    # Print results
    print("Query:", query)
    print("Top Matches:")
    for i in range(len(I[0])):
        print(f"- {documents[I[0][i]]} (score: {D[0][i]})")

    #for i in I[0]:
    #    print(f"- {documents[i]} (score: {D[0][i]})")



if __name__ == "__main__":
    main()

