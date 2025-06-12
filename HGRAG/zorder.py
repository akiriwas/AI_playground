import math
import random

def encode_morton(vector, max_bits_per_dimension):
    """
    Encodes an N-dimensional vector into a 1-dimensional Z-order (Morton) index.

    The Z-order curve interleaves the bits of the coordinates from each dimension.
    For example, for 2D (x, y) with 2 bits each:
    x = 10 (binary)  -> x_1 x_0
    y = 11 (binary)  -> y_1 y_0
    Morton code (LSB to MSB): x_0 y_0 x_1 y_1 = 0111 (binary) = 7 (decimal)
    Morton code (MSB to LSB): y_1 x_1 y_0 x_0 = 1101 (binary) = 13 (decimal)
    The function now constructs the Morton code as (y_1 x_1 y_0 x_0).

    Args:
        vector (list or tuple): An N-dimensional vector of non-negative integers.
                                Each element represents a coordinate in that dimension.
        max_bits_per_dimension (int): The maximum number of bits required to represent
                                      any coordinate in the vector. This determines the
                                      resolution of the space and the total length of
                                      the Morton code. For example, if your coordinates
                                      go up to 1023, you need max_bits_per_dimension=10
                                      (since 2^10 - 1 = 1023).

    Returns:
        int: The 1-dimensional Z-order index.

    Raises:
        ValueError: If any coordinate is negative or exceeds the maximum value
                    representable by max_bits_per_dimension.
    """
    num_dimensions = len(vector)
    morton_code = 0

    # Validate input coordinates
    max_coord_value = (1 << max_bits_per_dimension) - 1
    for i, coord in enumerate(vector):
        if not isinstance(coord, int) or coord < 0:
            raise ValueError(
                f"Coordinate at index {i} ({coord}) must be a non-negative integer."
            )
        if coord > max_coord_value:
            raise ValueError(
                f"Coordinate at index {i} ({coord}) exceeds the maximum "
                f"value allowed by max_bits_per_dimension ({max_bits_per_dimension} bits). "
                f"Max allowed value is {max_coord_value}."
            )

    # Iterate through each bit position of the original coordinates, from least significant to most significant.
    # This ensures that lower-order bits of coordinates are placed at lower-order positions in the Morton code.
    for bit_pos in range(max_bits_per_dimension):
        # For each bit position, interleave the bit from each dimension
        for dim_idx in range(num_dimensions):
            # Extract the bit at 'bit_pos' from the current dimension's coordinate.
            # (coord >> bit_pos) shifts the desired bit to the 0th position.
            # & 1 isolates that bit.
            bit = (vector[dim_idx] >> bit_pos) & 1
            # Calculate the target position for this bit within the final Morton code.
            # This position is determined by (current bit position in original coordinate * number of dimensions)
            # plus the index of the current dimension.
            target_morton_bit_pos = bit_pos * num_dimensions + dim_idx
            # Set the extracted bit in the calculated position within the morton_code.
            # `|=` (bitwise OR assignment) ensures other bits already set are not affected.
            morton_code |= (bit << target_morton_bit_pos)

    return morton_code

def decode_morton(morton_code, num_dimensions, max_bits_per_dimension):
    """
    Decodes a 1-dimensional Z-order (Morton) index back into an N-dimensional vector.

    This function reverses the bit interleaving process.

    Args:
        morton_code (int): The 1-dimensional Z-order index to decode.
        num_dimensions (int): The number of dimensions of the original vector.
        max_bits_per_dimension (int): The maximum number of bits used per dimension
                                      during the encoding process. This must be the
                                      same value as used for encoding.

    Returns:
        list: The decoded N-dimensional vector of integers.

    Raises:
        ValueError: If the morton_code is negative.
    """
    if not isinstance(morton_code, int) or morton_code < 0:
        raise ValueError("Morton code must be a non-negative integer.")
    if num_dimensions <= 0 or max_bits_per_dimension <= 0:
        raise ValueError("Number of dimensions and max_bits_per_dimension must be positive.")

    # Initialize the N-dimensional vector with zeros
    vector = [0] * num_dimensions

    # Iterate through each bit position of the Morton code, from least significant to most significant.
    # The total number of bits in the Morton code is max_bits_per_dimension * num_dimensions.
    total_morton_bits = max_bits_per_dimension * num_dimensions
    for bit_idx_in_morton in range(total_morton_bits):
        # Determine which dimension this bit belongs to.
        # The modulo operation (bit_idx_in_morton % num_dimensions) tells us which dimension's
        # turn it is to receive a bit.
        dim_to_set = bit_idx_in_morton % num_dimensions

        # Determine the position of this bit within its original dimension's coordinate.
        # The integer division (bit_idx_in_morton // num_dimensions) gives us the
        # bit position within that specific dimension's coordinate (0 for LSB, 1 for next, etc.).
        # We are building the coordinate from LSB to MSB.
        bit_pos_in_dim = bit_idx_in_morton // num_dimensions

        # Extract the bit from the Morton code at the current bit_idx_in_morton position.
        # (morton_code >> bit_idx_in_morton) shifts the desired bit to the 0th position.
        # & 1 isolates that bit.
        bit_value = (morton_code >> bit_idx_in_morton) & 1

        # Set the extracted bit in the correct position within the correct dimension's coordinate.
        # (bit_value << bit_pos_in_dim) places the bit at its correct position in the coordinate.
        # |= is a bitwise OR assignment, which effectively sets the bit without affecting others.
        vector[dim_to_set] |= (bit_value << bit_pos_in_dim)

    return vector

# --- Examples ---

# Example 1: 2D vector (common case)
print("--- 2D Example ---")
vector_2d = [2, 3]  # (x=2, y=3)
max_bits = 2        # Each coordinate up to 2^2 - 1 = 3
morton_2d = encode_morton(vector_2d, max_bits)
print(f"Original 2D vector: {vector_2d}")
print(f"Encoded Morton code: {morton_2d} (Binary: {bin(morton_2d)})")
decoded_2d = decode_morton(morton_2d, len(vector_2d), max_bits)
print(f"Decoded 2D vector: {decoded_2d}")
print(f"Matches original: {vector_2d == decoded_2d}\n")

# Example 2: 3D vector
print("--- 3D Example ---")
vector_3d = [4, 5, 6]  # (x=4, y=5, z=6)
max_bits = 3           # Each coordinate up to 2^3 - 1 = 7
morton_3d = encode_morton(vector_3d, max_bits)
print(f"Original 3D vector: {vector_3d}")
print(f"Encoded Morton code: {morton_3d} (Binary: {bin(morton_3d)})")
decoded_3d = decode_morton(morton_3d, len(vector_3d), max_bits)
print(f"Decoded 3D vector: {decoded_3d}")
print(f"Matches original: {vector_3d == decoded_3d}\n")

# Example 3: Higher dimensions (5D)
print("--- 5D Example ---")
vector_5d = [1, 0, 2, 0, 3]
max_bits = 2 # Each coordinate up to 3
morton_5d = encode_morton(vector_5d, max_bits)
print(f"Original 5D vector: {vector_5d}")
print(f"Encoded Morton code: {morton_5d} (Binary: {bin(morton_5d)})")
decoded_5d = decode_morton(morton_5d, len(vector_5d), max_bits)
print(f"Decoded 5D vector: {decoded_5d}")
print(f"Matches original: {vector_5d == decoded_5d}\n")

print("--- 128D Example ---")
vector_128d = [random.randint(0,5) for _ in range(128)]
max_bits = 3 # Each coordinate up to 3
morton_128d = encode_morton(vector_128d, max_bits)
print(f"Original 128D vector: {vector_128d}")
print(f"Encoded Morton code: {morton_128d} (Binary: {bin(morton_128d)})")
decoded_128d = decode_morton(morton_128d, len(vector_128d), max_bits)
print(f"Decoded 128D vector: {decoded_128d}")
print(f"Matches original: {vector_128d == decoded_128d}\n")



# Example 4: Coordinates with leading zeros (e.g., 0)
print("--- Zero Coordinates Example ---")
vector_zeros = [0, 0, 0]
max_bits = 4
morton_zeros = encode_morton(vector_zeros, max_bits)
print(f"Original vector (zeros): {vector_zeros}")
print(f"Encoded Morton code: {morton_zeros} (Binary: {bin(morton_zeros)})")
decoded_zeros = decode_morton(morton_zeros, len(vector_zeros), max_bits)
print(f"Decoded vector (zeros): {decoded_zeros}")
print(f"Matches original: {vector_zeros == decoded_zeros}\n")

# Example 5: Edge case - single dimension (N=1)
print("--- 1D Example ---")
vector_1d = [123]
max_bits = 8 # Enough to cover 123 (2^7 = 128)
morton_1d = encode_morton(vector_1d, max_bits)
print(f"Original 1D vector: {vector_1d}")
print(f"Encoded Morton code: {morton_1d} (Binary: {bin(morton_1d)})")
decoded_1d = decode_morton(morton_1d, len(vector_1d), max_bits)
print(f"Decoded 1D vector: {decoded_1d}")
print(f"Matches original: {vector_1d == decoded_1d}\n")

# Example 6: Error handling - coordinate too large
print("--- Error Handling Example (Coord Too Large) ---")
try:
    encode_morton([5], max_bits_per_dimension=2) # Max 3, 5 is too large
except ValueError as e:
    print(f"Error caught as expected: {e}\n")

# Example 7: Error handling - negative coordinate
print("--- Error Handling Example (Negative Coord) ---")
try:
    encode_morton([-1, 2], max_bits_per_dimension=3)
except ValueError as e:
    print(f"Error caught as expected: {e}\n")

# Example 8: Error handling - negative morton code
print("--- Error Handling Example (Negative Morton Code) ---")
try:
    decode_morton(-10, 2, 3)
except ValueError as e:
    print(f"Error caught as expected: {e}\n")

