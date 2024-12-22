import torch
from torch import Tensor
from typing import Optional, Tuple, Union


class DiagonalProjectionComposition(torch.nn.Module):
    """
    Implements the diagonal projection composition operation (⊙) for matrices.

    This operation performs a weighted matrix multiplication where the weights
    are determined by the distance of elements from the diagonal. This can be
    particularly efficient for matrices with important near-diagonal structure.

    Args:
        sigma (float): Controls the spread of the weight function. Larger values
            allow more influence from elements far from the diagonal.
        truncation_epsilon (float, optional): Threshold for truncating the weight
            function. Elements with weights below this value are ignored.
            Defaults to 1e-10.
        device (Union[str, torch.device], optional): Device to place the module on.
            Defaults to 'cuda' if available, else 'cpu'.

    Example:
        >>> op = DiagonalProjectionComposition(sigma=1.0)
        >>> A = torch.randn(3, 3)
        >>> B = torch.randn(3, 3)
        >>> C = op(A, B)  # Computes A⊙B
    """

    def __init__(
        self,
        sigma: float,
        truncation_epsilon: float = 1e-10,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.sigma = sigma
        self.truncation_epsilon = truncation_epsilon

        # Set device - use CUDA if available by default
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Register sigma as a buffer so it moves with the module
        self.register_buffer(
            "sigma_tensor", torch.tensor(sigma, device=self.device)
        )

    def _compute_weight_matrix(self, size: int) -> Tensor:
        """
        Computes the weight matrix based on distance from diagonal.

        Args:
            size (int): Size of the matrix (n for n×n matrix)

        Returns:
            Tensor: Weight matrix of shape (size, size)
        """
        indices = torch.arange(size, device=self.device)
        distances = (
            indices.unsqueeze(0) - indices.unsqueeze(1)
        ).abs()
        weights = torch.exp(
            -distances.pow(2) / (2 * self.sigma_tensor.pow(2))
        )

        # Apply truncation
        weights = weights * (weights > self.truncation_epsilon)
        return weights

    def _validate_inputs(self, A: Tensor, B: Tensor) -> None:
        """
        Validates input matrices for compatibility.

        Args:
            A (Tensor): First input matrix
            B (Tensor): Second input matrix

        Raises:
            ValueError: If matrices have incompatible dimensions
        """
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("Inputs must be 2-dimensional matrices")
        if A.size(1) != B.size(0):
            raise ValueError(
                f"Matrix dimensions incompatible: {A.size()} and {B.size()}"
            )

    def forward(self, A: Tensor, B: Tensor) -> Tensor:
        """
        Computes the diagonal projection composition A⊙B.

        Args:
            A (Tensor): First input matrix of shape (m, n)
            B (Tensor): Second input matrix of shape (n, p)

        Returns:
            Tensor: Result matrix of shape (m, p)

        Note:
            The operation is not associative: (A⊙B)⊙C ≠ A⊙(B⊙C)
        """
        self._validate_inputs(A, B)

        m, n = A.size()
        p = B.size(1)

        # Compute weight matrices for both dimensions
        W1 = self._compute_weight_matrix(n)

        # For each output element (i,j), compute weighted sum
        result = torch.zeros(m, p, device=self.device)

        # Optimize computation by pre-weighting matrices
        weighted_A = A.unsqueeze(2) * W1.unsqueeze(
            0
        )  # Shape: (m, n, 1)
        weighted_B = W1.unsqueeze(2) * B.t().unsqueeze(
            0
        )  # Shape: (n, p, 1)

        # Compute final result using batch matrix multiplication
        result = torch.bmm(
            weighted_A.transpose(0, 1), weighted_B.transpose(0, 1)
        ).sum(dim=0)

        return result

    def get_complexity(self, n: int) -> Tuple[int, float]:
        """
        Computes theoretical complexity for n×n matrices.

        Args:
            n (int): Matrix dimension

        Returns:
            Tuple[int, float]: (Number of non-zero weights, Theoretical FLOPS)
        """
        weights = self._compute_weight_matrix(n)
        nnz = torch.count_nonzero(weights).item()
        flops = (
            nnz * n
        )  # Multiplication and addition for each non-zero weight
        return nnz, float(flops)


# Example usage and testing
def run_example() -> None:
    """
    Demonstrates usage of the DiagonalProjectionComposition module.
    """
    # Create module instance
    op = DiagonalProjectionComposition(sigma=1.0)

    # Create example matrices
    A = torch.tensor(
        [[1.0, 0.5, 0.1], [0.5, 2.0, 0.5], [0.1, 0.5, 1.0]]
    )

    B = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    # Compute diagonal projection composition
    C = op(A, B)

    print("Input matrix A:")
    print(A)
    print("\nInput matrix B:")
    print(B)
    print("\nResult A⊙B:")
    print(C)

    # Demonstrate complexity analysis
    nnz, flops = op.get_complexity(3)
    print("\nComplexity analysis for 3×3 matrices:")
    print(f"Non-zero weights: {nnz}")
    print(f"Theoretical FLOPS: {flops}")


if __name__ == "__main__":
    run_example()
