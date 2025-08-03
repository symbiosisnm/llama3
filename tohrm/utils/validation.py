from typing import List, Dict
import torch

def validate_input(context: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    """
    Validate that the context tensor matches the expected shape.

    The first dimension (batch) may be flexible if the corresponding expected
    dimension is None.

    Args:
        context: The input tensor whose shape to validate.
        expected_shape: Tuple specifying expected dimensions; use None to
            denote a flexible dimension (e.g., batch size).

    Raises:
        AssertionError: If the shape does not match the expected specification.
    """
    actual_shape = context.shape
    if len(actual_shape) != len(expected_shape):
        raise AssertionError(f"Tensor rank {len(actual_shape)} != expected {len(expected_shape)}")
    for idx, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise AssertionError(
                f"Dimension {idx} mismatch: actual {actual_shape} != expected {expected_shape}"
            )


def check_convergence(trace: List[Dict], tol: float = 1e-2) -> bool:
    """
    Check whether latent states in the trace converge within a tolerance.

    Specifically, examines the norm of the 'l' (low-level) latent in each cycle
    and asserts that successive differences are below the given tolerance.

    Args:
        trace: List of trace entries produced by ReasoningController.run.
        tol: Tolerance for difference between successive norms.

    Returns:
        True if convergence criteria are met; False otherwise.
    """
    if not trace:
        return True
    norms = [t['l'].norm().item() for t in trace]
    for i in range(1, len(norms)):
        if abs(norms[i] - norms[i - 1]) > tol:
            return False
    return True
