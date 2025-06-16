import math
import random

def euclid_dist(vec_1, vec_2):
    if len(vec_1) != len(vec_2):
        return 0.0

    sum = 0.0
    for i in range(len(vec_1)):
        sum = sum + (vec_1[i] - vec_2[i])**2
    return sum ** 0.5


def encode_morton(vector, max_bits_per_dimension):
    """
    Encodes an N-dimensional vector of non-negative integers into a 1-dimensional
    Z-order (Morton) index.

    The Z-order curve interleaves the bits of the coordinates from each dimension.
    For example, for 2D (x, y) with 2 bits each:
    x = 10 (binary)  -> x_1 x_0
    y = 11 (binary)  -> y_1 y_0
    Morton code (LSB to MSB): x_0 y_0 x_1 y_1 = 0111 (binary) = 7 (decimal)

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
    Decodes a 1-dimensional Z-order (Morton) index back into an N-dimensional
    vector of non-negative integers.

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

def float_encode_morton(vector_float, max_bits_per_dimension):
    """
    Encodes an N-dimensional vector of floating-point numbers (range -1.0 to 1.0)
    into a 1-dimensional Z-order (Morton) index.

    This function scales and offsets the float values to non-negative integers
    before using the integer-based Morton encoding.

    Args:
        vector_float (list or tuple): An N-dimensional vector of floats, where
                                      each component is in the range [-1.0, 1.0].
        max_bits_per_dimension (int): The number of bits to use for representing
                                      each dimension's scaled integer value.
                                      Higher values provide more precision.

    Returns:
        int: The 1-dimensional Z-order index.

    Raises:
        ValueError: If any component is outside the [-1.0, 1.0] range or
                    if max_bits_per_dimension is not positive.
    """
    if not (isinstance(max_bits_per_dimension, int) and max_bits_per_dimension > 0):
        raise ValueError("max_bits_per_dimension must be a positive integer.")

    # Calculate the maximum integer value for the given number of bits
    # This determines the resolution of our integer grid
    max_int_value = (1 << max_bits_per_dimension) - 1

    integer_vector = []
    for i, f_val in enumerate(vector_float):
        if not (isinstance(f_val, (float, int)) and -1.0 <= f_val <= 1.0):
            raise ValueError(
                f"Vector component at index {i} ({f_val}) must be a float "
                f"or int in the range [-1.0, 1.0]."
            )
        # Scale f_val from [-1.0, 1.0] to [0.0, 1.0], then to [0, max_int_value]
        scaled_val = (f_val + 1.0) / 2.0 * max_int_value
        # Round to the nearest integer
        integer_vector.append(round(scaled_val))

    return encode_morton(integer_vector, max_bits_per_dimension)

def float_decode_morton(morton_code, num_dimensions, max_bits_per_dimension):
    """
    Decodes a 1-dimensional Z-order (Morton) index back into an N-dimensional
    vector of floating-point numbers in the range [-1.0, 1.0].

    This function uses the integer-based Morton decoding and then scales the
    integer coordinates back to floats.

    Args:
        morton_code (int): The 1-dimensional Z-order index to decode.
        num_dimensions (int): The number of dimensions of the original vector.
        max_bits_per_dimension (int): The number of bits used per dimension
                                      during the encoding process.

    Returns:
        list: The decoded N-dimensional vector of floats.

    Raises:
        ValueError: If the morton_code is negative or parameters are invalid.
    """
    if not (isinstance(num_dimensions, int) and num_dimensions > 0):
        raise ValueError("num_dimensions must be a positive integer.")
    if not (isinstance(max_bits_per_dimension, int) and max_bits_per_dimension > 0):
        raise ValueError("max_bits_per_dimension must be a positive integer.")

    integer_vector = decode_morton(morton_code, num_dimensions, max_bits_per_dimension)

    # Calculate the maximum integer value that was used for scaling
    max_int_value = (1 << max_bits_per_dimension) - 1

    float_vector = []
    for i_val in integer_vector:
        # Scale i_val from [0, max_int_value] to [0.0, 1.0], then to [-1.0, 1.0]
        # Handle potential division by zero if max_bits_per_dimension is 0 (though already guarded)
        if max_int_value == 0: # This case would imply max_bits_per_dimension is 0, which is already caught.
            float_vector.append(0.0)
        else:
            scaled_val = (i_val / max_int_value) * 2.0 - 1.0
            float_vector.append(scaled_val)

    return float_vector

# --- Examples for Integer Operations (Existing) ---
print("--- Integer Operations Examples ---")
# Example 1: 2D vector (common case)
vector_2d = [2, 3]  # (x=2, y=3)
max_bits = 2        # Each coordinate up to 2^2 - 1 = 3
morton_2d = encode_morton(vector_2d, max_bits)
print(f"Original 2D integer vector: {vector_2d}")
print(f"Encoded Morton code: {morton_2d} (Binary: {bin(morton_2d)})")
decoded_2d = decode_morton(morton_2d, len(vector_2d), max_bits)
print(f"Decoded 2D integer vector: {decoded_2d}")
print(f"Matches original: {vector_2d == decoded_2d}\n")

# Example 2: 3D vector
vector_3d = [4, 5, 6]  # (x=4, y=5, z=6)
max_bits = 3           # Each coordinate up to 2^3 - 1 = 7
morton_3d = encode_morton(vector_3d, max_bits)
print(f"Original 3D integer vector: {vector_3d}")
print(f"Encoded Morton code: {morton_3d} (Binary: {bin(morton_3d)})")
decoded_3d = decode_morton(morton_3d, len(vector_3d), max_bits)
print(f"Decoded 3D integer vector: {decoded_3d}")
print(f"Matches original: {vector_3d == decoded_3d}\n")

# Example 3: Higher dimensions (5D)
vector_5d = [1, 0, 2, 0, 3]
max_bits = 2 # Each coordinate up to 3
morton_5d = encode_morton(vector_5d, max_bits)
print(f"Original 5D integer vector: {vector_5d}")
print(f"Encoded Morton code: {morton_5d} (Binary: {bin(morton_5d)})")
decoded_5d = decode_morton(morton_5d, len(vector_5d), max_bits)
print(f"Decoded 5D integer vector: {decoded_5d}")
print(f"Matches original: {vector_5d == decoded_5d}\n")

# Example 4: Zero Coordinates
vector_zeros = [0, 0, 0]
max_bits = 4
morton_zeros = encode_morton(vector_zeros, max_bits)
print(f"Original integer vector (zeros): {vector_zeros}")
print(f"Encoded Morton code: {morton_zeros} (Binary: {bin(morton_zeros)})")
decoded_zeros = decode_morton(morton_zeros, len(vector_zeros), max_bits)
print(f"Decoded integer vector (zeros): {decoded_zeros}")
print(f"Matches original: {vector_zeros == decoded_zeros}\n")

# Example 5: 1D
vector_1d = [123]
max_bits = 8
morton_1d = encode_morton(vector_1d, max_bits)
print(f"Original 1D integer vector: {vector_1d}")
print(f"Encoded Morton code: {morton_1d} (Binary: {bin(morton_1d)})")
decoded_1d = decode_morton(morton_1d, len(vector_1d), max_bits)
print(f"Decoded 1D integer vector: {decoded_1d}")
print(f"Matches original: {vector_1d == decoded_1d}\n")

# --- New Examples for Floating-Point Operations ---
print("--- Floating-Point Operations Examples (Range: -1.0 to 1.0) ---")

# Example 6: 2D float vector
float_vec_2d = [0.5, -0.75]
float_max_bits = 10 # Higher bits for more precision (2^10 = 1024 distinct values)
float_morton_2d = float_encode_morton(float_vec_2d, float_max_bits)
print(f"Original 2D float vector: {float_vec_2d}")
print(f"Encoded Morton code: {float_morton_2d} (Binary: {bin(float_morton_2d)})")
float_decoded_2d = float_decode_morton(float_morton_2d, len(float_vec_2d), float_max_bits)
print(f"Decoded 2D float vector: {float_decoded_2d}")
# Note: Floating point comparisons need a tolerance
print(f"Matches original (approx): {[round(x, 5) for x in float_vec_2d] == [round(x, 5) for x in float_decoded_2d]}\n")


# Example 7: 3D float vector, including boundary values
float_vec_3d = [-1.0, 0.0, 1.0]
float_max_bits_3d = 8 # 2^8 = 256 distinct values
float_morton_3d = float_encode_morton(float_vec_3d, float_max_bits_3d)
print(f"Original 3D float vector: {float_vec_3d}")
print(f"Encoded Morton code: {float_morton_3d} (Binary: {bin(float_morton_3d)})")
float_decoded_3d = float_decode_morton(float_morton_3d, len(float_vec_3d), float_max_bits_3d)
print(f"Decoded 3D float vector: {float_decoded_3d}")
print(f"Matches original (approx): {[round(x, 5) for x in float_vec_3d] == [round(x, 5) for x in float_decoded_3d]}\n")

# Example 7.1: 128D float vector, including boundary values
float_vec_128d = [-1.0, 0.0, 1.0]
float_vec_128d = [random.uniform(-1.0, 1.0) for _ in range(768)]
float_max_bits_128d = 16 # 2^8 = 256 distinct values
float_morton_128d = float_encode_morton(float_vec_128d, float_max_bits_128d)
print(f"Original 128D float vector: {float_vec_128d}")
print(f"Encoded Morton code: {float_morton_128d} (Binary: {bin(float_morton_128d)})")
float_decoded_128d = float_decode_morton(float_morton_128d, len(float_vec_128d), float_max_bits_128d)
print(f"Decoded 128D float vector: {float_decoded_128d}")
print(f"Matches original (approx): {[round(x, 5) for x in float_vec_128d] == [round(x, 5) for x in float_decoded_128d]}")
print(f"Distance between vectors: {euclid_dist(float_vec_128d, float_decoded_128d)}\n")



# Example 8: Float vector with varying components and higher dimensions
float_vec_4d = [0.1, -0.2, 0.3, -0.4]
float_max_bits_4d = 12 # 2^12 = 4096 distinct values
float_morton_4d = float_encode_morton(float_vec_4d, float_max_bits_4d)
print(f"Original 4D float vector: {float_vec_4d}")
print(f"Encoded Morton code: {float_morton_4d} (Binary: {bin(float_morton_4d)})")
float_decoded_4d = float_decode_morton(float_morton_4d, len(float_vec_4d), float_max_bits_4d)
print(f"Decoded 4D float vector: {float_decoded_4d}")
print(f"Matches original (approx): {[round(x, 5) for x in float_vec_4d] == [round(x, 5) for x in float_decoded_4d]}\n")


# --- Error Handling Examples (Added for Float functions) ---
print("--- Error Handling Examples (Float functions) ---")

# Example 9: Float component out of range (> 1.0)
try:
    float_encode_morton([1.1, 0.0], 10)
except ValueError as e:
    print(f"Error caught as expected: {e}\n")

# Example 10: Float component out of range (< -1.0)
try:
    float_encode_morton([0.0, -1.5], 10)
except ValueError as e:
    print(f"Error caught as expected: {e}\n")

# Example 11: Invalid max_bits_per_dimension (float_encode)
try:
    float_encode_morton([0.5, 0.5], 0)
except ValueError as e:
    print(f"Error caught as expected: {e}\n")

# Example 12: Invalid num_dimensions (float_decode)
try:
    float_decode_morton(100, 0, 10)
except ValueError as e:
    print(f"Error caught as expected: {e}\n")


