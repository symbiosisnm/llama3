from typing import List, Dict
import torch

class ReasoningController:
    """
    Runs multi-cycle hierarchical reasoning with optional middle module.
    Tracks latent trajectories and provides convergence diagnostics.
    """
    def __init__(self, hmod, lmod, mmod=None, cycles: int = 4, max_l_steps: int = 10, tol: float = 1e-3):
        self.hmod = hmod
        self.lmod = lmod
        self.mmod = mmod
        self.cycles = cycles
        self.max_l_steps = max_l_steps
        self.tol = tol

    def run(
        self,
        context: torch.Tensor,
        h0: torch.Tensor | None = None,
        m0: torch.Tensor | None = None,
        l0: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, List[Dict], Dict]:
        """
        Execute hierarchical reasoning cycles.

        Args:
            context: Input tensor [B, ctx_dim].
            h0: Initial high-level state [B, h_hidden_dim].
            m0: Initial middle-level state [B, m_hidden_dim] if using MModule.
            l0: Initial low-level state [B, l_hidden_dim].
            verbose: Print norms each cycle.

        Returns:
            Tuple containing final H state, M state (or None), L state, trace list, and diagnostics dict.
        """
        B = context.shape[0]
        device = context.device
        h = h0 if h0 is not None else torch.zeros(B, self.hmod.hidden_dim, device=device)
        m = None
        if self.mmod is not None:
            m = m0 if m0 is not None else torch.zeros(B, self.mmod.hidden_dim, device=device)
        l = l0 if l0 is not None else torch.zeros(B, self.lmod.hidden_dim, device=device)

        trace: List[Dict] = []
        for c in range(self.cycles):
            # high-level update
            h, plan = self.hmod(h, context)
            # middle-level refinement if available
            if self.mmod is not None:
                m = self.mmod.converge(m, plan, max_steps=self.max_l_steps, tol=self.tol)
                refined_plan = m
            else:
                refined_plan = plan
            # low-level convergence
            l = self.lmod.converge(l, refined_plan, max_steps=self.max_l_steps, tol=self.tol)

            trace.append({
                'cycle': c,
                'h': h.detach().cpu(),
                'l': l.detach().cpu(),
                'plan': plan.detach().cpu(),
                'refined_plan': refined_plan.detach().cpu() if isinstance(refined_plan, torch.Tensor) else None
            })
            if verbose:
                print(f"Cycle {c}: H norm={h.norm().item():.3f}, L norm={l.norm().item():.3f}")

        diagnostics = self._validate_trace(trace)
        return h, m, l, trace, diagnostics

    def _validate_trace(self, trace: List[Dict]) -> Dict:
        """
        Compute simple convergence diagnostics from the trace.

        Args:
            trace: List of dictionaries containing latents per cycle.

        Returns:
            Dictionary with norms per cycle and a stability flag.
        """
        norms = [{'h': t['h'].norm().item(), 'l': t['l'].norm().item()} for t in trace]
        stable = all(abs(norms[i]['l'] - norms[i-1]['l']) < 1e-2 for i in range(1, len(norms)))
        return {'norms': norms, 'stable': stable}
