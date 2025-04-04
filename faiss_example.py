import faiss
import numpy as np

d = 256  # Dimension of vectors
nb = 1000000  # Number of vectors in the dataset

# Generate random data
np.random.seed(42)
database_vectors = np.random.random((nb, d)).astype('float32')

# Generate a query vector
query_vector = np.random.random((1, d)).astype('float32')

index = faiss.IndexFlatL2(d)  # L2 (Euclidean distance) index
print("Is trained:", index.is_trained)  # Should be True for Flat index

index.add(database_vectors)  # Add vectors to the index
print("Total vectors in index:", index.ntotal)  # Should be 1000

k = 5  # Number of nearest neighbors
distances, indices = index.search(query_vector, k)

print("Nearest neighbor indices:", indices)
print("Nearest neighbor distances:", distances)

