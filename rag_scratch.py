
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

DEBUG = 1

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

# Assumes that these are vector lists.
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

def main():
    #Put the embeddings AND the appropriate text into a list
    vector_db = [(doc, generate_embedding(doc)) for doc in documents]

    # Query Example
    query = "Tell me about the Moon landing"
    query_embedding = generate_embedding(query)
    dprint(str(len(query_embedding)) + ": " + str(query_embedding))

    similarities = []
    for doc, embedding in vector_db:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((doc, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("Query:", query)
    print("Top Matches:")
    ret = similarities[:2]
    for doc, similarity in similarities[:2]:
        print(f"- {doc} (score: {similarity})")



if __name__ == "__main__":
    main()

