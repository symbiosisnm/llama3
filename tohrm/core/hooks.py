# Stubs for optional retrieval and reward integration.

class RetrievalHook:
    def retrieve(self, query: str):
        """Return list of facts/documents given a query."""
        # TODO: implement retrieval mechanism
        raise NotImplementedError

class RewardHook:
    def score(self, h_latent, l_latent) -> float:
        """Return reward signal for RLHF or adaptive halting."""
        # TODO: compute reward based on latent states
        raise NotImplementedError
