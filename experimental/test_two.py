# import torch
# import torch.nn as nn
# from typing import Tuple, Optional
# import math

# class FastMatrixDecomp(torch.autograd.Function):
#     """
#     A novel matrix decomposition operation that combines aspects of LU decomposition
#     and Strassen's algorithm to potentially speed up certain linear algebra operations.

#     The core idea is to recursively decompose matrices into smaller sub-matrices while
#     maintaining useful mathematical properties that allow for faster downstream computations.
#     """

#     @staticmethod
#     def forward(ctx, input: torch.Tensor, block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the decomposition.

#         Args:
#             input: Input matrix of shape (batch_size, M, N)
#             block_size: Size of blocks for recursive decomposition

#         Returns:
#             Tuple of three tensors representing the decomposed components:
#             - U: Upper triangular factor with modified diagonal elements
#             - D: Diagonal scaling factor
#             - L: Lower triangular factor with special structure
#         """
#         assert input.dim() >= 2, "Input must be at least 2-dimensional"
#         batch_size = input.size(0) if input.dim() > 2 else 1
#         M, N = input.size(-2), input.size(-1)

#         # Reshape input if necessary
#         if input.dim() == 2:
#             input = input.unsqueeze(0)

#         # Initialize output tensors
#         U = torch.zeros_like(input)
#         D = torch.zeros(batch_size, min(M, N), device=input.device)
#         L = torch.zeros_like(input)

#         def recursive_decomp(matrix: torch.Tensor, start_m: int, start_n: int, size: int):
#             """
#             Recursively decompose the matrix into smaller blocks.
#             """
#             if size <= block_size:
#                 # Base case: perform direct computation on small block
#                 block = matrix[..., start_m:start_m+size, start_n:start_n+size]

#                 # Novel transformation: combine aspects of LU with special scaling
#                 diag = torch.diagonal(block, dim1=-2, dim2=-1)
#                 scale = torch.sqrt(torch.abs(diag) + 1e-6)

#                 # Modified decomposition that preserves certain matrix properties
#                 u_block = torch.triu(block / scale.unsqueeze(-2))
#                 l_block = torch.tril(block / scale.unsqueeze(-1), diagonal=-1) + torch.eye(size, device=block.device)

#                 # Store results
#                 U[..., start_m:start_m+size, start_n:start_n+size] = u_block
#                 D[..., start_m:start_m+size] = scale
#                 L[..., start_m:start_m+size, start_n:start_n+size] = l_block
#             else:
#                 # Recursive case: split into smaller blocks
#                 new_size = size // 2
#                 recursive_decomp(matrix, start_m, start_n, new_size)
#                 recursive_decomp(matrix, start_m + new_size, start_n + new_size, new_size)

#                 # Novel cross-block optimization
#                 if start_m + new_size < M and start_n + new_size < N:
#                     cross_block = matrix[..., start_m:start_m+new_size, start_n+new_size:start_n+2*new_size]
#                     # Apply special transformation to cross-block
#                     transformed = cross_block * torch.exp(-torch.norm(cross_block, dim=-1, keepdim=True))
#                     U[..., start_m:start_m+new_size, start_n+new_size:start_n+2*new_size] = transformed

#         # Start recursive decomposition
#         min_dim = min(M, N)
#         recursive_size = 2 ** math.floor(math.log2(min_dim))
#         recursive_decomp(input, 0, 0, recursive_size)

#         # Save tensors for backward pass
#         ctx.save_for_backward(input, U, D, L)
#         return U, D, L

#     @staticmethod
#     def backward(ctx, grad_U: torch.Tensor, grad_D: torch.Tensor, grad_L: torch.Tensor) -> Tuple[torch.Tensor, None]:
#         """
#         Backward pass for gradient computation.
#         """
#         input, U, D, L = ctx.saved_tensors

#         # Combine gradients from all components
#         grad_input = grad_U.clone()
#         grad_input += torch.diag_embed(grad_D)
#         grad_input += grad_L

#         return grad_input, None

# def apply_fast_decomp(input: torch.Tensor, block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Convenience function to apply the fast matrix decomposition.

#     Args:
#         input: Input tensor of shape (batch_size, M, N)
#         block_size: Size of blocks for recursive decomposition

#     Returns:
#         Tuple of (U, D, L) tensors representing the decomposed matrix
#     """
#     return FastMatrixDecomp.apply(input, block_size)

# # Example usage and testing
# def test_decomposition():
#     """
#     Test the decomposition on a sample matrix.
#     """
#     # Create a test matrix
#     input_matrix = torch.randn(2, 64, 64)

#     # Apply decomposition
#     U, D, L = apply_fast_decomp(input_matrix)

#     # Reconstruct original matrix
#     reconstructed = U * D.unsqueeze(-1) @ L

#     # Check reconstruction error
#     error = torch.norm(input_matrix - reconstructed)
#     print(f"Reconstruction error: {error:.6f}")

#     return U, D, L
from typing import Optional, Tuple
import torch
from loguru import logger
import math


class AdaptiveBlockMatMul(torch.autograd.Function):
    """
    A custom PyTorch operation implementing an adaptive block matrix multiplication algorithm.

    This operation automatically determines optimal block sizes based on matrix dimensions
    and hardware characteristics, then performs block-wise multiplication with parallel
    execution where possible. Special care is taken to maintain numerical precision
    through accumulated operations.
    """

    @staticmethod
    def _determine_optimal_block_size(
        m: int, n: int, k: int
    ) -> Tuple[int, int, int]:
        """
        Determines the optimal block size based on matrix dimensions and L2 cache size.

        Args:
            m: First matrix rows
            n: Second matrix columns
            k: First matrix columns/Second matrix rows

        Returns:
            Tuple of block sizes (m_block, n_block, k_block)
        """
        # Estimate L2 cache size (typical size on modern CPUs)
        l2_cache = 256 * 1024  # 256KB
        element_size = 4  # float32

        # Calculate maximum elements that can fit in L2 cache
        # We want (m_block * k_block + k_block * n_block + m_block * n_block) * element_size < l2_cache
        total_elements = l2_cache // element_size

        # More conservative block size to reduce accumulation error
        block_size = int(
            math.sqrt(total_elements / 6)
        )  # Reduced from /3 to /6 for better precision

        # Adjust block size to matrix dimensions
        block_size = min(block_size, min(m, n, k))

        # Ensure block size is at least 16 for efficiency
        block_size = max(16, block_size)

        return block_size, block_size, block_size

    @staticmethod
    def forward(
        ctx, matrix1: torch.Tensor, matrix2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass implementing block matrix multiplication with enhanced precision.

        Args:
            ctx: Context object for backward pass
            matrix1: First input matrix (m x k)
            matrix2: Second input matrix (k x n)

        Returns:
            Result matrix (m x n)
        """
        logger.debug("Starting adaptive block matrix multiplication")

        assert (
            matrix1.dim() == 2 and matrix2.dim() == 2
        ), "Input matrices must be 2-dimensional"
        assert matrix1.size(1) == matrix2.size(
            0
        ), "Matrix dimensions must match for multiplication"

        m, k = matrix1.size()
        k, n = matrix2.size()

        # Store inputs for backward pass
        ctx.save_for_backward(matrix1, matrix2)

        # Get optimal block sizes
        m_block, n_block, k_block = (
            AdaptiveBlockMatMul._determine_optimal_block_size(m, n, k)
        )
        logger.info(
            f"Using block sizes: {m_block} x {n_block} x {k_block}"
        )

        # Initialize result matrix with higher precision
        result = torch.zeros(
            m, n, device=matrix1.device, dtype=torch.float64
        )

        # Perform block matrix multiplication with enhanced precision
        for i in range(0, m, m_block):
            i_end = min(i + m_block, m)
            for j in range(0, n, n_block):
                j_end = min(j + n_block, n)
                block_sum = torch.zeros(
                    (i_end - i, j_end - j),
                    device=matrix1.device,
                    dtype=torch.float64,
                )

                for k_start in range(0, k, k_block):
                    k_end = min(k_start + k_block, k)

                    # Extract blocks and convert to higher precision
                    block1 = matrix1[i:i_end, k_start:k_end].to(
                        torch.float64
                    )
                    block2 = matrix2[k_start:k_end, j:j_end].to(
                        torch.float64
                    )

                    # Multiply blocks and accumulate with higher precision
                    block_sum.addmm_(block1, block2)

                result[i:i_end, j:j_end] = block_sum

        # Convert back to input precision
        result = result.to(matrix1.dtype)

        logger.debug("Completed block matrix multiplication")
        return result

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Backward pass computing gradients for block matrix multiplication.

        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of the loss with respect to the output

        Returns:
            Tuple of gradients for each input tensor
        """
        matrix1, matrix2 = ctx.saved_tensors

        grad1 = None
        grad2 = None

        if ctx.needs_input_grad[0]:
            # Gradient for first matrix: grad_output @ matrix2.T
            grad1 = AdaptiveBlockMatMul.apply(
                grad_output, matrix2.t()
            )

        if ctx.needs_input_grad[1]:
            # Gradient for second matrix: matrix1.T @ grad_output
            grad2 = AdaptiveBlockMatMul.apply(
                matrix1.t(), grad_output
            )

        return grad1, grad2


def adaptive_matmul(
    matrix1: torch.Tensor, matrix2: torch.Tensor
) -> torch.Tensor:
    """
    Functional interface for adaptive block matrix multiplication.

    This function automatically determines optimal block sizes and performs
    block-wise matrix multiplication that can be more cache-efficient than
    standard matrix multiplication for certain matrix sizes.

    Args:
        matrix1: First input matrix (m x k)
        matrix2: Second input matrix (k x n)

    Returns:
        Result of matrix multiplication (m x n)

    Example:
        >>> A = torch.randn(1000, 800)
        >>> B = torch.randn(800, 1200)
        >>> C = adaptive_matmul(A, B)  # Returns 1000 x 1200 matrix
    """
    return AdaptiveBlockMatMul.apply(matrix1, matrix2)


# Add comprehensive test cases
if __name__ == "__main__":
    # Configure logging
    logger.add("adaptive_matmul.log", rotation="500 MB")

    # Test small matrices
    logger.info("Testing small matrix multiplication")
    A = torch.randn(100, 80)
    B = torch.randn(80, 120)

    # Compare results with standard matmul
    result_standard = torch.mm(A, B)
    result_adaptive = adaptive_matmul(A, B)

    # Verify correctness with stricter tolerance
    assert torch.allclose(
        result_standard, result_adaptive, rtol=1e-7, atol=1e-7
    ), "Results don't match!"
    logger.success("Small matrix multiplication test passed")

    # Test larger matrices
    logger.info("Testing large matrix multiplication")
    A = torch.randn(2000, 1500)
    B = torch.randn(1500, 2500)

    result_standard = torch.mm(A, B)
    result_adaptive = adaptive_matmul(A, B)

    # Verify correctness with appropriate tolerance for larger matrices
    assert torch.allclose(
        result_standard, result_adaptive, rtol=1e-6, atol=1e-6
    ), "Results don't match!"
    logger.success("Large matrix multiplication test passed")

    # Test extreme aspect ratios
    logger.info("Testing matrices with extreme aspect ratios")
    A = torch.randn(5000, 50)
    B = torch.randn(50, 5000)

    result_standard = torch.mm(A, B)
    result_adaptive = adaptive_matmul(A, B)

    assert torch.allclose(
        result_standard, result_adaptive, rtol=1e-6, atol=1e-6
    ), "Results don't match!"
    logger.success("Extreme aspect ratio test passed")
