from HSFC_KNN import MultiHSFC_KNN

import numpy as np
import random
import math
import argparse
import time

def BruteForceKNN(vectors, query, k):
    """
    Finds the K nearest neighbors to a query vector using Euclidean distance.

    Args:
        vectors (list of lists/tuples): A list of N-dimensional vectors. Each inner
                                        list/tuple represents a vector.
        query (list/tuple): A single N-dimensional query vector.
        k (int): The number of nearest neighbors to return.

    Returns:
        list: A list of indices (integers) corresponding to the K nearest
              vectors in the original 'vectors' list.
    """
    if not vectors:
        print("Error: The 'vectors' list cannot be empty.")
        return []
    if k <= 0:
        print("Error: 'k' must be a positive integer.")
        return []
    if k > len(vectors):
        print(f"Warning: 'k' ({k}) is greater than the number of available vectors ({len(vectors)}). "
              "Returning all available vector indices sorted by distance.")
        k = len(vectors)

    distances = []
    # Iterate through each vector in the 'vectors' list along with its index
    for i, vector in enumerate(vectors):
        # Ensure dimensions match for valid Euclidean distance calculation
        if len(vector) != len(query):
            print(f"Warning: Vector at index {i} has a different dimension "
                  f"({len(vector)}) than the query vector ({len(query)}). Skipping this vector.")
            continue

        # Calculate Euclidean distance
        # The sum of squared differences between corresponding elements
        squared_diff_sum = sum((v - q)**2 for v, q in zip(vector, query))
        # The square root of the sum gives the Euclidean distance
        distance = math.sqrt(squared_diff_sum)

        # Store the distance and the original index of the vector
        distances.append((distance, i))

    # Sort the distances list based on the distance value (first element of the tuple)
    # This will arrange the vectors from nearest to farthest
    distances.sort(key=lambda x: x[0])

    # Extract the indices of the K smallest distances
    # We take the second element of the tuple (which is the index) for the first 'k' elements
    k_nearest_indices = [idx for dist, idx in distances[:k]]

    return k_nearest_indices

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-C', '--curves', type=int, help='Number of Hilbert Curves')
    parser.add_argument('-O', '--order', type=int, help='Complexity order of Hilbert Curves')

    args = parser.parse_args()

    arg_num_curves = args.curves
    arg_order = args.order


    vectors = fvecs_read("../datasets/siftsmall/siftsmall_base.fvecs")
    lowest = vectors[0][0]
    highest = vectors[0][0]
    for v in vectors:
        for x in v:
            if x < lowest:
                lowest = x
            if x > highest:
                highest = x
    print(f"lowest = {lowest}, highest = {highest}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    print(vectors)
    vectors = [v.tolist() for v in vectors]

    comp_n_dims = 128

    multi_knn = MultiHSFC_KNN(n_dims=comp_n_dims, p_order=arg_order, num_curves=arg_num_curves, max_candidates_per_curve=10)
    ingest_start_time = time.time()
    multi_knn.ingest_dataset(vectors)
    ingest_end_time = time.time()

    # Look at a random vector and offset it a little bit as a query
    chosen_idx = random.randint(0,len(vectors)-1)
    query_vec = vectors[chosen_idx]
    print(f"Chosen Query Index = {chosen_idx}")
    print(f"Chosen Query Vector = {vectors[chosen_idx]}")
    #print(f"Chosen Query Vector Perturbed = {query_vec}")


    query_start_time = time.time()
    multi_neighbors_idxs = multi_knn.query(query_vec, 10)
    query_end_time = time.time()
    print("MultiHSFC_KNN Neighbors:")
    print(multi_neighbors_idxs)

    true_neighbors_idxs = BruteForceKNN(vectors, query_vec, 10)
    print("True KNN Neighbors:")
    print(true_neighbors_idxs)

    recall = list(set(multi_neighbors_idxs).intersection(set(true_neighbors_idxs)))
    recall_score = len(recall)/len(true_neighbors_idxs)
    print(f"RESULTS -- O: {arg_order}, C: {arg_num_curves}, R: {recall_score}, IT: {ingest_end_time-ingest_start_time:.6f}, QT: {query_end_time-query_start_time:.6f}")

    #for idx in multi_neighbors_idxs:
    #    print(f"  {np.array(vectors[idx])}")
    #for n in multi_neighbors: print(f"  {np.array(n)}")



if __name__ == "__main__":
    main()
