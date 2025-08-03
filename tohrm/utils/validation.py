# Validation helpers.
from torch import Tensor


def validate_input(context: Tensor, expected_shape):
    """Assert that context tensor matches the expected shape (batch dimension may be flexible)."""
    # TODO: compare shape dimensions with allowance for unspecified batch size
    pass


def check_convergence(trace, tol: float = 1e-2):
    """Examine latent norms in trace and ensure successive differences are below tolerance."""
    # TODO: iterate through trace and compute norm differences
    pass
