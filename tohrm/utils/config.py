# Typed configuration for dimensions and controller parameters.
from dataclasses import dataclass
from typing import Optional

@dataclass
class HRMConfig:
    context_dim: int = 128
    h_hidden_dim: int = 64
    h_plan_dim: int = 32
    m_hidden_dim: Optional[int] = None
    m_plan_dim: Optional[int] = None
    l_hidden_dim: int = 64
    cycles: int = 4
    max_l_steps: int = 10
    max_m_steps: int = 5
    tol: float = 1e-3
    use_middle: bool = False
    adaptive_halting: bool = False
    dropout_rate: float = 0.1
    use_reward_hook: bool = False
    use_retrieval_hook: bool = False
    reward_threshold: Optional[float] = None
    max_episodes: Optional[int] = None
