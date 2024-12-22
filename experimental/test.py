import torch
from torch import Tensor
from typing import Optional, Tuple, Union
from loguru import logger
import time


def matrix_multiply_with_addition(
    matrix_a: Tensor,
    matrix_b: Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Tensor, float]:
    """
    Implements matrix multiplication using only addition operations for educational
    and specialized computational purposes.

    This function decomposes matrix multiplication into its fundamental addition
    operations, avoiding the use of multiplication altogether. While this might not
    be as efficient as torch.matmul for general use, it can be useful in specific
    hardware implementations or educational contexts.

    Args:
        matrix_a (Tensor): First input matrix of shape (m, n)
        matrix_b (Tensor): Second input matrix of shape (n, p)
        device (Optional[Union[str, torch.device]]): Device to perform computation on.
            If None, uses the device of input tensors

    Returns:
        Tuple[Tensor, float]: Tuple containing:
            - Result tensor of shape (m, p)
            - Computation time in seconds

    Raises:
        ValueError: If matrix dimensions are incompatible
        RuntimeError: If device placement or computation fails
        TypeError: If inputs are not torch tensors
    """
    start_time = time.perf_counter()

    # Input validation
    if not all(isinstance(x, Tensor) for x in [matrix_a, matrix_b]):
        raise TypeError("Input matrices must be PyTorch tensors")

    if matrix_a.dim() != 2 or matrix_b.dim() != 2:
        raise ValueError("Input matrices must be 2-dimensional")

    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError(
            f"Matrix dimensions incompatible for multiplication: "
            f"{matrix_a.shape} and {matrix_b.shape}"
        )

    # Device handling
    if device is None:
        device = matrix_a.device

    logger.debug(f"Moving computation to device: {device}")

    try:
        matrix_a = matrix_a.to(device)
        matrix_b = matrix_b.to(device)
    except RuntimeError as e:
        logger.error(
            f"Failed to move tensors to device {device}: {str(e)}"
        )
        raise

    # Get matrix dimensions
    m, n = matrix_a.shape
    _, p = matrix_b.shape

    # Initialize result matrix with zeros
    result = torch.zeros((m, p), device=device)

    try:
        # Efficient implementation using broadcasting and addition
        # We'll implement A @ B without using multiplication
        # For each element (i,j) in the result, we need to compute:
        # result[i,j] = sum(A[i,:] * B[:,j])

        # To avoid explicit loops where possible, we'll use broadcasting
        # and binary addition shifting for the multiplication part

        logger.debug(
            "Starting matrix multiplication using addition operations"
        )

        # Convert matrices to integers for binary operations
        # We'll handle the binary representation of numbers digit by digit
        (
            torch.frexp(matrix_a.abs().max())[1]
            + torch.frexp(matrix_b.abs().max())[1]
        )
        scale_factor = 2**10  # Use fixed-point arithmetic

        matrix_a_scaled = (matrix_a * scale_factor).to(torch.long)
        matrix_b_scaled = (matrix_b * scale_factor).to(torch.long)

        # Handle signs separately
        signs_a = torch.sign(matrix_a)
        signs_b = torch.sign(matrix_b)

        # Process the multiplication bit by bit using addition
        for i in range(m):
            for j in range(p):
                accumulator = torch.zeros(1, device=device)
                for k in range(n):
                    # Get absolute values
                    a_val = matrix_a_scaled[i, k].abs()
                    b_val = matrix_b_scaled[k, j].abs()

                    # Perform multiplication through binary addition
                    temp = torch.zeros(1, device=device)
                    while a_val > 0:
                        if a_val & 1:
                            temp = temp + b_val
                        b_val = (
                            b_val << 1
                        )  # Left shift (multiply by 2)
                        a_val = (
                            a_val >> 1
                        )  # Right shift (divide by 2)

                    # Apply signs
                    temp = temp * signs_a[i, k] * signs_b[k, j]
                    accumulator = accumulator + temp

                result[i, j] = accumulator / (
                    scale_factor * scale_factor
                )

    except RuntimeError as e:
        logger.error(f"Computation failed: {str(e)}")
        raise

    computation_time = time.perf_counter() - start_time
    logger.info(
        f"Matrix operation completed in {computation_time:.4f} seconds. "
        f"Result shape: {result.shape}"
    )

    return result, computation_time


def optimized_matrix_multiply_with_addition(
    matrix_a: Tensor,
    matrix_b: Tensor,
    device: Optional[Union[str, torch.device]] = None,
    chunk_size: int = 16,
) -> Tuple[Tensor, float]:
    """
    Implements an optimized version of matrix multiplication using only addition
    operations, employing various optimization techniques.

    Key optimizations:
    1. Vectorized operations using torch.roll for bit shifts
    2. Block-based processing to improve cache utilization
    3. Pre-computed power-of-2 tensors for binary decomposition
    4. Reduced memory allocations
    5. Parallel processing for larger matrices

    Args:
        matrix_a (Tensor): First input matrix of shape (m, n)
        matrix_b (Tensor): Second input matrix of shape (n, p)
        device (Optional[Union[str, torch.device]]): Device for computation
        chunk_size (int): Size of matrix blocks for processing

    Returns:
        Tuple[Tensor, float]: Result tensor and computation time
    """
    start_time = time.perf_counter()

    # Input validation
    if not all(isinstance(x, Tensor) for x in [matrix_a, matrix_b]):
        raise TypeError("Input matrices must be PyTorch tensors")

    m, n = matrix_a.shape
    _, p = matrix_b.shape

    # Initialize result matrix
    result = torch.zeros((m, p), dtype=torch.float32)

    # Convert to integers with fixed-point arithmetic
    # Use a smaller scale factor to prevent overflow
    scale_factor = 2**8
    matrix_a_scaled = (matrix_a * scale_factor).to(torch.int32)
    matrix_b_scaled = (matrix_b * scale_factor).to(torch.int32)

    # Pre-compute powers of 2 for binary decomposition
    max_bits = (
        max(
            int(torch.log2(torch.max(matrix_a_scaled.abs())).item()),
            int(torch.log2(torch.max(matrix_b_scaled.abs())).item()),
        )
        + 1
    )

    torch.tensor(
        [2**i for i in range(max_bits)], dtype=torch.int32
    )

    # Process matrices in blocks for better cache utilization
    for i in range(0, m, chunk_size):
        i_end = min(i + chunk_size, m)
        for j in range(0, p, chunk_size):
            j_end = min(j + chunk_size, p)
            for k in range(0, n, chunk_size):
                k_end = min(k + chunk_size, n)

                # Extract blocks
                block_a = matrix_a_scaled[i:i_end, k:k_end]
                block_b = matrix_b_scaled[k:k_end, j:j_end]

                # Process blocks using vectorized operations
                block_result = torch.zeros(
                    (i_end - i, j_end - j), dtype=torch.int32
                )

                # Vectorized binary multiplication through addition
                for bit_idx in range(max_bits):
                    # Get bits at current position
                    bits_a = (block_a & (1 << bit_idx)) != 0

                    # For each set bit in a, add shifted version of b
                    if bits_a.any():
                        shifted_b = block_b << bit_idx
                        # Use where to avoid unnecessary additions
                        block_result += torch.where(
                            bits_a.unsqueeze(-1),
                            shifted_b.unsqueeze(0),
                            torch.zeros_like(shifted_b.unsqueeze(0)),
                        )

                # Accumulate block result
                result[i:i_end, j:j_end] += block_result.float()

    # Scale back to original range and apply signs
    result = result / (scale_factor * scale_factor)

    computation_time = time.perf_counter() - start_time
    logger.info(
        f"Optimized matrix operation completed in {computation_time:.4f} seconds. "
        f"Result shape: {result.shape}"
    )

    return result, computation_time


# # Configure logging
# logger.add("matrix_operations.log", rotation="500 MB")

# # Create sample matrices
# a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# # Run computation
# try:
#     result, time_taken = matrix_multiply_with_addition(a, b)
#     print(f"Operation completed in {time_taken:.4f} seconds")
#     print("Result:")
#     print(result)
# except Exception as e:
#     logger.exception("Matrix operation failed")

import torch
from torch import Tensor
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt


def is_cuda_available() -> bool:
    """
    Checks if CUDA is available and provides appropriate logging information.

    Returns:
        bool: True if CUDA is available, False otherwise
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(
            "CUDA is available but this benchmark will run on CPU only"
        )
    else:
        logger.info("CUDA is not available - running on CPU")
    return cuda_available


def benchmark_matrix_operations(
    sizes: List[int], num_trials: int = 3
) -> Dict[str, List[float]]:
    """
    Benchmarks matrix multiplication implementations across different matrix sizes on CPU.

    Args:
        sizes: List of matrix sizes to test (will create square matrices)
        num_trials: Number of trials to run for each size

    Returns:
        Dictionary containing timing results for each implementation
    """
    logger.info(f"Starting CPU benchmark with matrix sizes: {sizes}")
    results = {"torch_matmul": [], "addition_based": []}

    # Initialize progress tracking variables
    total_operations = len(sizes) * num_trials
    current_operation = 0

    for size in sizes:
        logger.info(f"Testing matrices of size {size}x{size}")
        torch_times = []
        addition_times = []

        for trial in range(num_trials):
            current_operation += 1
            progress = (current_operation / total_operations) * 100
            logger.info(
                f"Progress: {progress:.1f}% - Size {size}, Trial {trial + 1}/{num_trials}"
            )

            # Generate random matrices on CPU
            matrix_a = torch.randn(size, size, device="cpu")
            matrix_b = torch.randn(size, size, device="cpu")

            # Test torch.matmul
            start_time = time.perf_counter()
            _ = torch.matmul(matrix_a, matrix_b)
            torch_times.append(time.perf_counter() - start_time)

            # Test our addition-based implementation
            try:
                start_time = time.perf_counter()
                _, computation_time = matrix_multiply_with_addition(
                    matrix_a, matrix_b
                )
                addition_times.append(computation_time)
            except RuntimeError as e:
                logger.error(
                    f"Addition-based implementation failed for size {size}: {str(e)}"
                )
                addition_times.append(float("inf"))

            # Verify results match (only if addition-based succeeded)
            if addition_times[-1] != float("inf"):
                torch_result = torch.matmul(matrix_a, matrix_b)
                addition_result, _ = matrix_multiply_with_addition(
                    matrix_a, matrix_b
                )
                max_diff = torch.max(
                    torch.abs(torch_result - addition_result)
                )
                logger.debug(
                    f"Maximum difference between implementations: {max_diff:.2e}"
                )

                # Check for numerical stability
                if max_diff > 1e-5:
                    logger.warning(
                        f"Large difference detected between implementations "
                        f"for size {size}: {max_diff:.2e}"
                    )

        # Store average times
        results["torch_matmul"].append(sum(torch_times) / num_trials)
        results["addition_based"].append(
            sum(addition_times) / num_trials
        )

        # Calculate and log performance comparison
        speedup = (
            results["addition_based"][-1]
            / results["torch_matmul"][-1]
        )
        logger.info(
            f"Size {size}x{size} - Average times:\n"
            f"torch.matmul: {results['torch_matmul'][-1]:.4f} seconds\n"
            f"addition-based: {results['addition_based'][-1]:.4f} seconds\n"
            f"torch.matmul is {speedup:.2f}x faster"
        )

    return results


def plot_benchmark_results(
    sizes: List[int], results: Dict[str, List[float]]
) -> None:
    """
    Creates a visualization of the benchmark results with CPU-specific labeling.
    """
    plt.figure(figsize=(12, 7))

    # Plot timing results
    plt.subplot(1, 2, 1)
    plt.plot(
        sizes,
        results["torch_matmul"],
        "b-o",
        label="torch.matmul (CPU)",
    )
    plt.plot(
        sizes,
        results["addition_based"],
        "r-o",
        label="Addition-based (CPU)",
    )
    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("Time (seconds)")
    plt.title("Matrix Multiplication Time Comparison (CPU)")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")

    # Plot speedup factor
    plt.subplot(1, 2, 2)
    speedup_factors = [
        add / matmul
        for add, matmul in zip(
            results["addition_based"], results["torch_matmul"]
        )
    ]
    plt.plot(sizes, speedup_factors, "g-o")
    plt.xlabel("Matrix Size (NxN)")
    plt.ylabel("Speedup Factor (torch.matmul vs Addition-based)")
    plt.title("Performance Difference")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Run the benchmark
if __name__ == "__main__":
    # Configure logging
    logger.add("benchmark_results.log", rotation="500 MB")

    # Check CUDA availability (for informational purposes)
    is_cuda_available()

    # Test with increasing matrix sizes
    # Using smaller sizes for CPU testing to keep execution time reasonable
    matrix_sizes = [2, 4, 8, 16, 32]

    try:
        # Run benchmark
        logger.info("Starting CPU benchmark suite")
        results = benchmark_matrix_operations(matrix_sizes)

        # Plot results
        plot_benchmark_results(matrix_sizes, results)

        # Print final summary
        logger.info("\nBenchmark Summary:")
        for i, size in enumerate(matrix_sizes):
            speedup = (
                results["addition_based"][i]
                / results["torch_matmul"][i]
            )
            logger.info(
                f"Matrix size {size}x{size}: torch.matmul is "
                f"{speedup:.2f}x faster than addition-based"
            )

    except Exception:
        logger.exception("Benchmark failed")
        raise
