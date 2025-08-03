import torch
import torch.nn as nn
import torch.nn.functional as F

class HModule(nn.Module):
    """High-level abstract planner; slow, global latent."""
    def __init__(self, input_dim: int, hidden_dim: int, plan_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, plan_dim)
        self.hidden_dim = hidden_dim
        self.plan_dim = plan_dim

    def forward(self, h_prev: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the next high-level hidden state and plan.

        Args:
            h_prev: [B, hidden_dim] previous hidden state
            context: [B, input_dim] contextual input

        Returns:
            A tuple of (new hidden state, plan vector)
        """
        h_new = self.rnn(context, h_prev)
        plan = self.proj(h_new)
        return h_new, plan

class LModule(nn.Module):
    """Low-level executor that performs rapid, local refinements."""
    def __init__(self, plan_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRUCell(plan_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def step(self, l_prev: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """Perform one GRU-based refinement step."""
        return self.rnn(plan, l_prev)

    def converge(
        self,
        l_init: torch.Tensor,
        plan: torch.Tensor,
        max_steps: int = 10,
        tol: float = 1e-3,
    ) -> torch.Tensor:
        """
        Iterate until the low-level hidden state converges to a fixed point.

        Args:
            l_init: Initial low-level state [B, hidden_dim].
            plan: Plan vector from higher module [B, plan_dim].
            max_steps: Maximum number of refinement iterations.
            tol: Convergence tolerance for mean L2 norm.

        Returns:
            Final low-level hidden state.
        """
        l = l_init
        for _ in range(max_steps):
            l_next = self.step(l, plan)
            if torch.norm(l_next - l, p=2, dim=-1).mean() < tol:
                l = l_next
                break
            l = l_next
        return l

class MModule(nn.Module):
    """Optional middle module for multi-level hierarchy."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRUCell(in_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim

    def converge(
        self,
        m_init: torch.Tensor,
        plan: torch.Tensor,
        max_steps: int = 5,
        tol: float = 1e-3,
    ) -> torch.Tensor:
        """
        Iteratively refine an intermediate hidden state and project to a plan.

        Args:
            m_init: Initial hidden state for middle module [B, hidden_dim].
            plan: Incoming plan vector [B, in_dim].
            max_steps: Maximum number of refinement iterations.
            tol: Convergence tolerance for mean L2 norm.

        Returns:
            Refined plan vector [B, out_dim].
        """
        m = m_init
        for _ in range(max_steps):
            m_next = self.rnn(plan, m)
            if torch.norm(m_next - m, p=2, dim=-1).mean() < tol:
                m = m_next
                break
            m = m_next
        return self.proj(m)
