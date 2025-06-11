from numba import cuda

# CUDA Kernel!

@cuda.jit
def square_matrix_kernel(matrix, result):
    # Calculate the row and column index for each thread
    row, col = cuda.grid(2)

    # Check if the thread's indixes are within the bounds of the matrix
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # perform the square operation
        result[row, col] = matrix[row, col] ** 2


# Example usage
import numpy as np

# Create a sample matrix
matrix = np.array([[1,2,3], [4,5,6]], dtype=np.float32)

# Allocate memory on the device
d_matrix = cuda.to_device(matrix)
d_result = cuda.device_array(matrix.shape, dtype=np.float32)

# Configure the blocks
threads_per_block = (16, 16)
blocks_per_grid_x = int(np.ceil(matrix.shape[0] / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(matrix.shape[1] / threads_per_block[1]))
blocks_per_grid = (block_per_grid_x, blocks_per_grid_y)

# Launch the kernel
square_matrix_kernel[blocks_per_grid, threads_per_block](d_matrix, d_result)

# Copy the result back to the host
result = d_result.copy_to_host()

# Result is now in the host's result array

print("Printing the matrix: \n")
print(matrix)
print("Printing the result: \n")
print(result)
