import numpy as np
import random
import bisect
import math # Import the standard math module
from itertools import permutations # Used for generating all permutations for smaller n_dims

# Moved _HilbertMapper to top-level for reuse by both HSFC_KNN and MultiHSFC_KNN
class _HilbertMapper:
    """
    Internal helper class to handle the Hilbert curve mapping logic.
    Implements coordinates_to_int without external library.
    Based on a generalized bit manipulation algorithm for Hilbert curves.

    NOTE: Implementing a robust, general N-dimensional Hilbert curve
    mapping from scratch is a complex task. This implementation provides
    a simplified bit-wise transformation that aims to capture the essence
    of the Hilbert curve mapping. For applications requiring extreme
    precision or handling very high dimensions/orders, a more rigorous
    algorithm (e.g., based on Skilling's method or lookup tables)
    or a specialized library might be necessary.
    """
    def __init__(self, p_order: int, n_dims: int):
        self.p_order = p_order
        self.n_dims = n_dims

        # Precompute constants for transformations
        self._max_hilbert_index = (1 << (p_order * n_dims)) - 1
        self._max_coord_val = (1 << p_order) - 1

    def coordinates_to_int(self, coords: list[int]) -> int:
        """
        Converts N-dimensional integer coordinates to a 1-dimensional Hilbert index.
        The coordinates are assumed to be in the range [0, 2^p_order - 1].

        Args:
            coords (list[int]): N-dimensional integer coordinates.

        Returns:
            int: The 1-dimensional Hilbert index.

        Raises:
            ValueError: If the coordinate dimensions do not match n_dims.
            ValueError: If any coordinate is out of the valid range.
        """
        if len(coords) != self.n_dims:
            raise ValueError(f"Coordinate dimension ({len(coords)}) does not match "
                             f"initialized n_dims ({self.n_dims}).")

        # Check if coordinates are within the valid range
        for coord in coords:
            if not (0 <= coord <= self._max_coord_val):
                raise ValueError(f"Coordinate {coord} out of range [0, {self._max_coord_val}] for p_order {self.p_order}.")

        hilbert_index = 0
        state = 0 # This `state` variable accumulates transformations for the current Hilbert segment.

        # Iterate from the most significant bit to the least significant bit
        for i in range(self.p_order - 1, -1, -1):
            # Extract the i-th bit from each coordinate
            current_bits = [(c >> i) & 1 for c in coords]

            # Form a 'quadrant index' from these bits (like a Z-order within this bit plane)
            quadrant_index = sum(bit << dim_idx for dim_idx, bit in enumerate(current_bits))

            # Apply the Hilbert transformation based on the current 'state' and 'quadrant_index'.
            # `transformed_quadrant_index` is the actual value that contributes to the Hilbert index.
            transformed_quadrant_index = self._transform_quadrant(quadrant_index, state)

            # Append the transformed quadrant bits to the Hilbert index
            hilbert_index = (hilbert_index << self.n_dims) | transformed_quadrant_index

            # Update the state for the next iteration (next lower bit plane)
            state = self._update_state(quadrant_index, state)

        return hilbert_index

    def _transform_quadrant(self, quadrant_index: int, state: int) -> int:
        """
        Transforms a quadrant index based on the current Hilbert curve state.
        This is a critical part of the Hilbert curve algorithm.
        """
        j = 0
        if self.n_dims > 0 and state > 0:
            while j < self.n_dims and not ((state >> j) & 1):
                j += 1

        rotated_quadrant = ((quadrant_index >> j) | (quadrant_index << (self.n_dims - j))) & ((1 << self.n_dims) - 1)
        transformed_quadrant = rotated_quadrant ^ (state >> j)
        transformed_quadrant = transformed_quadrant ^ (state & ((1 << self.n_dims) -1))

        return transformed_quadrant


    def _update_state(self, quadrant_index: int, prev_state: int) -> int:
        """
        Updates the Hilbert curve state for the next bit plane.
        """
        new_state = prev_state ^ quadrant_index
        return new_state

class HSFC_KNN:
    """
    Implements an approximate K-nearest neighbor (KNN) algorithm using a
    single Hilbert Space-Filling Curve (HSFC).

    The HSFC maps N-dimensional points to a 1-dimensional index, allowing for
    approximate nearest neighbor searches by finding neighbors in the 1D space.

    Assumptions:
    - Input N-dimensional vectors have coordinates that are floats
      and are normalized to be within the range [-1.0, 1.0]. If your data is not
      within this range, you should normalize it before ingesting.
    - The `p_order` (order of the Hilbert curve) determines the resolution.
      Higher `p_order` means higher precision but larger 1D indices.
    """

    def __init__(self, n_dims: int, p_order: int):
        """
        Initializes the HSFC_KNN object.

        Args:
            n_dims (int): The number of dimensions of the vectors.
            p_order (int): The order of the Hilbert curve. This determines the
                           resolution of the 1D mapping. A point (x, y, ...)
                           in N-dimensions is mapped to a coordinate system
                           where each dimension ranges from 0 to 2^p_order - 1.
        """
        if n_dims <= 0 or p_order <= 0:
            raise ValueError("n_dims and p_order must be positive integers.")

        self.n_dims = n_dims
        self.p_order = p_order
        self.hilbert_mapper = _HilbertMapper(p_order, n_dims) # Use the external _HilbertMapper
        # self.data stores tuples of (hilbert_index, original_vector, original_index)
        self.data = []
        self._max_coord_val = (2 ** self.p_order) - 1

    def _vector_to_hilbert_coords(self, vector: list[float]) -> list[int]:
        """
        Internal helper function to scale an N-dimensional vector's
        coordinates to the Hilbert curve's integer coordinate system.

        It assumes the input vector's coordinates are in [-1.0, 1.0] range.
        It first shifts and scales the range to [0.0, 1.0] and then
        to the Hilbert curve's integer range [0, 2^p_order - 1].

        Args:
            vector (list[float]): An N-dimensional vector with float coordinates.

        Returns:
            list[int]: A list of integer coordinates suitable for the Hilbert curve.

        Raises:
            ValueError: If the vector dimensions do not match n_dims or if
                        any coordinate is outside the [-1.0, 1.0] range.
        """
        if len(vector) != self.n_dims:
            raise ValueError(
                f"Vector dimension ({len(vector)}) does not match "
                f"initialized n_dims ({self.n_dims})."
            )

        scaled_coords = []
        for coord in vector:
            if not (-1.0 <= coord <= 1.0):
                raise ValueError(
                    "Vector coordinates must be normalized to the range [-1.0, 1.0]."
                )
            shifted_coord = coord + 1.0 # Range [0.0, 2.0]
            normalized_coord = shifted_coord / 2.0 # Range [0.0, 1.0]
            scaled_coords.append(int(round(normalized_coord * self._max_coord_val)))
        return scaled_coords

    def ingest_dataset(self, vectors: list[list[float]]):
        """
        Ingests a list of N-dimensional vectors into the HSFC_KNN model.
        For each vector, it computes its 1-dimensional Hilbert index and stores
        the original vector along with its index. The internal data structure
        is then sorted by the Hilbert index.

        Args:
            vectors (list[list[float]]): A list of N-dimensional vectors.
                                         Each inner list is a single vector.
                                         Assumed to have coordinates in [-1.0, 1.0].

        Returns:
            None
        Raises:
            ValueError: If any vector's dimension doesn't match n_dims or
                        if coordinates are not in the [-1.0, 1.0] range.
        """
        if not isinstance(vectors, list):
            raise TypeError("Input 'vectors' must be a list.")
        if not all(isinstance(v, list) for v in vectors):
            raise TypeError("Each item in 'vectors' must be a list (a vector).")

        self.data = []
        for original_index, original_vector in enumerate(vectors): # Added enumerate to get original index
            scaled_coords = self._vector_to_hilbert_coords(original_vector)
            hilbert_index = self.hilbert_mapper.coordinates_to_int(scaled_coords)
            self.data.append((hilbert_index, original_vector, original_index)) # Store original_index

        self.data.sort(key=lambda x: x[0])
        print(f"Ingested {len(self.data)} vectors.")

    def query(self, query_vector: list[float], k: int) -> list[int]: # Return type changed to list[int]
        """
        Finds the K-nearest neighbors to a query vector using the Hilbert index.

        The query vector is converted to its 1-dimensional Hilbert index.
        Then, the K-nearest indices in the stored dataset are found, and their
        corresponding original N-dimensional vectors are returned.

        Args:
            query_vector (list[float]): The N-dimensional vector to query for.
                                        Assumed to have coordinates in [-1.0, 1.0].
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            list[int]: A list of the original indices of the K-nearest N-dimensional vectors.

        Raises:
            ValueError: If k is not positive, exceeds the dataset size,
                        or if the query vector dimensions do not match n_dims.
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not self.data:
            print("Dataset is empty. Cannot perform query.")
            return []
        if k > len(self.data):
            print(f"Warning: k ({k}) is greater than the number of available "
                  f"data points ({len(self.data)}). Returning all available points.")
            k = len(self.data)

        scaled_query_coords = self._vector_to_hilbert_coords(query_vector)
        query_hilbert_index = self.hilbert_mapper.coordinates_to_int(scaled_query_coords)

        hilbert_indices_only = [item[0] for item in self.data]
        idx = bisect.bisect_left(hilbert_indices_only, query_hilbert_index)

        nearest_candidates = []
        left = idx - 1
        right = idx

        # Expand outwards from the insertion point to find k candidates
        # We collect 'sample_multiplier * k' candidates to have enough points to pick the closest K after sorting.
        # This provides a buffer, but for approximate KNN, more candidates might be sampled
        # to increase the chance of finding true nearest neighbors.
        sample_multiplier = 3 # Increased sampling
        max_samples = min(len(self.data), k * sample_multiplier)

        while len(nearest_candidates) < max_samples and (left >= 0 or right < len(self.data)):
            if left >= 0 and (right >= len(self.data) or \
                              abs(self.data[left][0] - query_hilbert_index) <= \
                              abs(self.data[right][0] - query_hilbert_index)):
                nearest_candidates.append(self.data[left])
                left -= 1
            elif right < len(self.data):
                nearest_candidates.append(self.data[right])
                right += 1
            else:
                break

        sorted_candidates = sorted(nearest_candidates,
                                   key=lambda x: abs(x[0] - query_hilbert_index))

        # Extract only the original indices for the top K
        result_indices = [idx for hilbert_idx, vec, idx in sorted_candidates[:k]]

        print(f"Query for {query_vector[:5]}... (Hilbert index: {query_hilbert_index}) "
              f"found {len(result_indices)} nearest neighbor indices.")
        return result_indices


class MultiHSFC_KNN:
    """
    Implements an approximate K-nearest neighbor (KNN) algorithm using multiple
    Hilbert Space-Filling Curves (HSFCs) with different dimension permutations.

    This approach aims to improve the accuracy of approximate KNN by using
    redundant indices, hoping to catch true neighbors that might be poorly
    mapped by a single Hilbert curve.

    Assumptions:
    - Input N-dimensional vectors have coordinates that are floats
      and are normalized to be within the range [-1.0, 1.0].
    - The `p_order` (order of the Hilbert curve) determines the resolution.
    """

    def __init__(self, n_dims: int, p_order: int, num_curves: int = 4, max_candidates_per_curve: int = 10):
        """
        Initializes the MultiHSFC_KNN object with multiple Hilbert curves.

        Args:
            n_dims (int): The number of dimensions of the vectors.
            p_order (int): The order of the Hilbert curve for all curves.
            num_curves (int): The number of different Hilbert curves to use.
                              Each curve will use a unique dimension permutation.
            max_candidates_per_curve (int): The maximum number of 1D nearest
                                            neighbors to retrieve from each
                                            individual Hilbert curve during a query.
                                            These are then re-ranked by true Euclidean distance.
        """
        if n_dims <= 0 or p_order <= 0 or num_curves <= 0 or max_candidates_per_curve <= 0:
            raise ValueError("All parameters must be positive integers.")
        
        # Factorial can grow extremely fast; limit calculation for small n_dims
        if n_dims < 10: # Heuristic limit; 9! is 362,880, 10! is 3,628,800
            # Use math.factorial instead of np.math.factorial
            max_unique_perms = math.factorial(n_dims)
            if num_curves > max_unique_perms:
                print(f"Warning: num_curves ({num_curves}) is greater than the number of unique "
                      f"permutations for {n_dims} dimensions ({max_unique_perms}). "
                      f"Using {max_unique_perms} unique permutations instead.")
                num_curves = max_unique_perms

        self.n_dims = n_dims
        self.p_order = p_order
        self.num_curves = num_curves
        self.max_candidates_per_curve = max_candidates_per_curve

        self.hilbert_mappers = []
        self.dimension_permutations = self._generate_unique_permutations(n_dims, num_curves)

        for _ in range(self.num_curves):
            self.hilbert_mappers.append(_HilbertMapper(p_order, n_dims))

        # self.data will be a list of lists.
        # self.data[i] stores the sorted (hilbert_index, original_vector, original_index) pairs for curve i.
        self.data = [[] for _ in range(self.num_curves)]
        self._max_coord_val = (2 ** self.p_order) - 1

    def _generate_unique_permutations(self, n: int, count: int) -> list[list[int]]:
        """
        Generates 'count' unique random permutations of dimensions [0, ..., n-1].
        If count is greater than n!, it generates all unique permutations.
        """
        if count <= 0:
            return []

        all_dims = list(range(n))
        if n < 10: # For small n, generate all permutations and then sample
            from itertools import permutations
            all_perms = list(permutations(all_dims))
            if count >= len(all_perms):
                return [list(p) for p in all_perms]
            else:
                return random.sample([list(p) for p in all_perms], count)
        else: # For larger n, random sampling is usually sufficient if count is reasonable
              # Generating all permutations is prohibitive.
            unique_perms = set()
            while len(unique_perms) < count:
                perm = tuple(random.sample(all_dims, n)) # random.sample generates unique elements
                unique_perms.add(perm)
            return [list(p) for p in unique_perms]

    def _vector_to_hilbert_coords(self, vector: list[float], permutation: list[int]) -> list[int]:
        """
        Internal helper function to scale an N-dimensional vector's
        coordinates to the Hilbert curve's integer coordinate system,
        applying a dimension permutation first.

        Assumes vector coordinates are in [-1.0, 1.0] range.
        """
        if len(vector) != self.n_dims:
            raise ValueError(
                f"Vector dimension ({len(vector)}) does not match "
                f"initialized n_dims ({self.n_dims})."
            )

        # Apply permutation first
        permuted_vector = [vector[permutation[i]] for i in range(self.n_dims)]

        scaled_coords = []
        for coord in permuted_vector:
            if not (-1.0 <= coord <= 1.0):
                raise ValueError(
                    "Vector coordinates must be normalized to the range [-1.0, 1.0]."
                )
            shifted_coord = coord + 1.0 # Range [0.0, 2.0]
            normalized_coord = shifted_coord / 2.0 # Range [0.0, 1.0]
            scaled_coords.append(int(round(normalized_coord * self._max_coord_val)))
        return scaled_coords

    def ingest_dataset(self, vectors: list[list[float]]):
        """
        Ingests a list of N-dimensional vectors into the MultiHSFC_KNN model.
        For each vector, it computes multiple 1-dimensional Hilbert indices
        (one for each internal Hilbert curve/permutation) and stores the
        original vector along with its index in the respective sorted lists.

        Args:
            vectors (list[list[float]]): A list of N-dimensional vectors.
                                         Each inner list is a single vector.
                                         Assumed to have coordinates in [-1.0, 1.0].
        Returns:
            None
        """
        if not isinstance(vectors, list):
            raise TypeError("Input 'vectors' must be a list.")
        if not all(isinstance(v, list) for v in vectors):
            raise TypeError("Each item in 'vectors' must be a list (a vector).")

        self.data = [[] for _ in range(self.num_curves)] # Reset data for ingestion

        for original_index, original_vector in enumerate(vectors): # Added enumerate
            for curve_idx in range(self.num_curves):
                permutation = self.dimension_permutations[curve_idx]
                scaled_coords = self._vector_to_hilbert_coords(original_vector, permutation)
                hilbert_index = self.hilbert_mappers[curve_idx].coordinates_to_int(scaled_coords)
                # Store original_index along with hilbert_index and original_vector
                self.data[curve_idx].append((hilbert_index, original_vector, original_index))

        for curve_idx in range(self.num_curves):
            self.data[curve_idx].sort(key=lambda x: x[0])
        print(f"Ingested {len(vectors)} vectors across {self.num_curves} Hilbert curves.")

    def query(self, query_vector: list[float], k: int) -> list[int]: # Return type changed to list[int]
        """
        Finds the K-nearest neighbors to a query vector using multiple Hilbert indices.

        The query vector is converted to multiple 1-dimensional Hilbert indices.
        Candidates are collected from each Hilbert curve and then re-ranked by
        their actual N-dimensional Euclidean distance.

        Args:
            query_vector (list[float]): The N-dimensional vector to query for.
                                        Assumed to have coordinates in [-1.0, 1.0].
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            list[int]: A list of the original indices of the K-nearest N-dimensional vectors (by Euclidean distance).
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if not self.data or all(not curve_data for curve_data in self.data):
            print("Dataset is empty. Cannot perform query.")
            return []

        # Use a dictionary to store unique candidate vectors along with their original index
        # This allows us to map back to the original index after re-ranking.
        # Key: tuple of vector (for hashing), Value: (original_vector as list, original_index)
        candidate_map = {} 

        for curve_idx in range(self.num_curves):
            if not self.data[curve_idx]: # Skip if this specific curve has no data
                continue

            permutation = self.dimension_permutations[curve_idx]
            scaled_query_coords = self._vector_to_hilbert_coords(query_vector, permutation)
            query_hilbert_index = self.hilbert_mappers[curve_idx].coordinates_to_int(scaled_query_coords)

            hilbert_indices_only = [item[0] for item in self.data[curve_idx]]
            idx = bisect.bisect_left(hilbert_indices_only, query_hilbert_index)

            # Expand outwards to find candidates
            collected_candidates = []
            
            # Simple expansion: collect a fixed number of elements around `idx`
            left = idx - 1
            right = idx
            
            # Ensure we don't go out of bounds or collect more than max_candidates_per_curve
            while len(collected_candidates) < self.max_candidates_per_curve and (left >= 0 or right < len(self.data[curve_idx])):
                if left >= 0 and (right >= len(self.data[curve_idx]) or \
                                  abs(self.data[curve_idx][left][0] - query_hilbert_index) <= \
                                  abs(self.data[curve_idx][right][0] - query_hilbert_index)):
                    collected_candidates.append(self.data[curve_idx][left])
                    left -= 1
                elif right < len(self.data[curve_idx]):
                    collected_candidates.append(self.data[curve_idx][right])
                    right += 1
                else:
                    break
            
            # Add unique candidates to the map
            for _, original_vec, original_idx in collected_candidates:
                candidate_map[tuple(original_vec)] = (original_vec, original_idx)

        if not candidate_map:
            print("No candidates found across all Hilbert curves.")
            return []

        # Convert candidates from map values to a list of (original_vec, original_idx) for re-ranking
        candidate_list_with_indices = list(candidate_map.values())

        # Re-rank candidates based on actual Euclidean distance
        distances_to_candidates = []
        for candidate_vec, original_idx in candidate_list_with_indices:
            dist = np.linalg.norm(np.array(query_vector) - np.array(candidate_vec))
            distances_to_candidates.append((dist, original_idx)) # Store (distance, original_index)

        distances_to_candidates.sort(key=lambda x: x[0])

        # Return top k original indices
        result_indices = [original_idx for dist, original_idx in distances_to_candidates[:k]]

        print(f"Query for {query_vector[:5]}... found {len(result_indices)} nearest neighbor indices "
              f"from {len(candidate_map)} unique candidates collected across {self.num_curves} curves.")
        return result_indices


# --- Example Usage (updated to return indices) ---
if __name__ == "__main__":
    # Example 1: 2D Vectors with [-1.0, 1.0] range using MultiHSFC_KNN
    print("--- Example 1: 2D Vectors (Range [-1.0, 1.0]) using MultiHSFC_KNN ---")
    
    multi_knn_2d = MultiHSFC_KNN(n_dims=2, p_order=4, num_curves=4, max_candidates_per_curve=5)

    dataset_2d = [
        [-0.9, -0.9], # 0
        [0.9, 0.9],   # 1
        [0.0, 0.0],   # 2
        [-0.8, -0.9], # 3
        [-0.9, -0.8], # 4
        [0.1, 0.05],  # 5
        [0.05, 0.1],  # 6
        [-1.0, -1.0], # 7
        [1.0, 1.0],   # 8
        [-0.6, 0.6],  # 9
        [0.6, -0.6],  # 10
        [0.1, 0.1],   # 11
        [0.15, 0.1],  # 12
        [0.1, 0.15],  # 13
        [0.2, 0.8]    # 14
    ]
    multi_knn_2d.ingest_dataset(dataset_2d)

    query_2d_vec = [-0.85, -0.88]
    k_val = 3
    nearest_2d_indices = multi_knn_2d.query(query_2d_vec, k_val)

    print(f"\nQuery vector: {query_2d_vec}")
    print(f"K = {k_val}")
    print("Nearest 2D neighbor ORIGINAL INDICES (approximate) using MultiHSFC_KNN:")
    for idx in nearest_2d_indices:
        print(f"  Index: {idx}, Vector: {dataset_2d[idx]}")

    print("\n" + "="*80 + "\n")

    # Example 2: 3D Vectors with [-1.0, 1.0] range using MultiHSFC_KNN
    print("--- Example 2: 3D Vectors (Range [-1.0, 1.0]) using MultiHSFC_KNN ---")
    multi_knn_3d = MultiHSFC_KNN(n_dims=3, p_order=3, num_curves=6, max_candidates_per_curve=10)

    dataset_3d = [
        [-0.9, -0.9, -0.9], # 0
        [0.9, 0.9, 0.9],    # 1
        [0.0, 0.0, 0.0],    # 2
        [-0.8, -0.9, -0.9], # 3
        [-0.9, -0.8, -0.9], # 4
        [-0.9, -0.9, -0.8], # 5
        [0.1, 0.05, 0.0],   # 6
        [0.05, 0.1, 0.0],   # 7
        [0.0, 0.05, 0.1],   # 8
        [-1.0, -1.0, -1.0], # 9
        [1.0, 1.0, 1.0],    # 10
        [-0.6, 0.6, -0.4],  # 11
        [0.6, -0.6, 0.4],   # 12
        [0.1, 0.1, 0.1],    # 13
        [0.15, 0.1, 0.1],   # 14
        [0.1, 0.15, 0.1]    # 15
    ]
    multi_knn_3d.ingest_dataset(dataset_3d)

    query_3d_vec = [-0.87, -0.89, -0.9]
    k_val_3d = 2
    nearest_3d_indices = multi_knn_3d.query(query_3d_vec, k_val_3d)

    print(f"\nQuery vector: {query_3d_vec}")
    print(f"K = {k_val_3d}")
    print("Nearest 3D neighbor ORIGINAL INDICES (approximate) using MultiHSFC_KNN:")
    for idx in nearest_3d_indices:
        print(f"  Index: {idx}, Vector: {dataset_3d[idx]}")

    print("\n" + "="*80 + "\n")

    # Example 3: Compare Single HSFC_KNN vs. MultiHSFC_KNN on a challenging point (returning indices)
    print("--- Comparison: Single HSFC_KNN vs. MultiHSFC_KNN (returning indices) ---")
    comp_n_dims = 2
    comp_p_order = 3 # Low p_order to highlight approximation errors
    comp_k_neighbors = 3

    comp_dataset = [
        [0.1, 0.1],       # 0
        [0.11, 0.12],     # 1
        [0.13, 0.1],      # 2
        [0.1, 0.13],      # 3
        [0.5, 0.5],       # 4
        [0.51, 0.52],     # 5
        [0.53, 0.5],      # 6
        [0.5, 0.53],      # 7
        [-0.9, -0.9],     # 8
        [-0.91, -0.92],   # 9
        [-0.93, -0.9],    # 10
        [-0.9, -0.93],    # 11
        [0.8, -0.8],      # 12
        [0.81, -0.82],    # 13
        [0.83, -0.8],     # 14
        [0.8, -0.83]      # 15
    ]
    comp_query_vec = [0.105, 0.108] # A point that might be tricky for a single curve

    # Single HSFC_KNN
    print("\nSingle HSFC_KNN:")
    single_knn = HSFC_KNN(n_dims=comp_n_dims, p_order=comp_p_order)
    single_knn.ingest_dataset(comp_dataset)
    single_indices = single_knn.query(comp_query_vec, comp_k_neighbors)
    print("Single HSFC_KNN Neighbor Indices:")
    for idx in single_indices: print(f"  Index: {idx}, Vector: {comp_dataset[idx]}")
    
    # Brute-force for comparison (for understanding ground truth)
    brute_neighbors_with_dist = []
    for idx, vec in enumerate(comp_dataset):
        dist = np.linalg.norm(np.array(comp_query_vec) - np.array(vec))
        brute_neighbors_with_dist.append((dist, idx, vec)) # Store (dist, original_idx, vec)
    brute_neighbors_with_dist.sort(key=lambda x: x[0])
    true_k_indices = [idx for dist, idx, vec in brute_neighbors_with_dist[:comp_k_neighbors]]
    print("\nTrue Brute-Force Neighbor Indices:")
    for idx in true_k_indices: print(f"  Index: {idx}, Vector: {comp_dataset[idx]}")


    # Multi HSFC_KNN
    print("\nMultiHSFC_KNN (num_curves=2, max_candidates_per_curve=5):")
    multi_knn = MultiHSFC_KNN(n_dims=comp_n_dims, p_order=comp_p_order, num_curves=2, max_candidates_per_curve=5)
    multi_knn.ingest_dataset(comp_dataset)
    multi_indices = multi_knn.query(comp_query_vec, comp_k_neighbors)
    print("MultiHSFC_KNN Neighbor Indices:")
    for idx in multi_indices: print(f"  Index: {idx}, Vector: {comp_dataset[idx]}")

