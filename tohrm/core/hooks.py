# Hooks for optional retrieval and reward integration.
from typing import List, Tuple, Any

class RetrievalHook:
    """
    Simple retrieval hook that can search over a provided collection of (key, value) pairs.

    Args:
        corpus: A mapping of keys to values representing facts or documents.
    """
    def __init__(self, corpus: dict[str, Any] | None = None):
        self.corpus = corpus or {}

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, Any]]:
        """
        Return a list of (key, value) pairs whose keys or values contain the query substring.

        This baseline implementation performs a simple case-insensitive substring search.
        For production use, replace with vector search or fuzzy matching.
        """
        query_lower = query.lower()
        results: List[Tuple[str, Any]] = []
        for key, value in self.corpus.items():
            try:
                if query_lower in str(key).lower() or query_lower in str(value).lower():
                    results.append((key, value))
                    if len(results) >= top_k:
                        break
            except Exception:
                # Ensure retrieval does not crash if value is not stringifiable
                continue
        return results

class RewardHook:
    """
    Simple reward function for RLHF or adaptive halting.

    This example computes a negative magnitude reward based on the norms of
    high-level and low-level latents, encouraging the model to converge to
    smaller activations. More sophisticated functions could incorporate
    task-specific scoring.
    """
    def score(self, h_latent, l_latent) -> float:
        # Compute L2 norms; detach to avoid autograd interfering
        try:
            h_norm = float(h_latent.norm().item())
        except Exception:
            h_norm = 0.0
        try:
            l_norm = float(l_latent.norm().item())
        except Exception:
            l_norm = 0.0
        return -(h_norm + l_norm)
