import pickle
import datetime
from typing import Any, Dict, List

class WorkingMemory:
    """Temporary key-value storage for intermediate reasoning results."""
    def __init__(self) -> None:
        self.slots: Dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Store a value under a key."""
        self.slots[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve a value by key, or None if absent."""
        return self.slots.get(key)

    def clear(self) -> None:
        """Clear all stored values."""
        self.slots.clear()
            def retrieve_prefix(self, prefix: str) -> Dict[str, Any]:
        """Return key-value pairs where the key starts with the given prefix."""
        return {k: v for k, v in self.slots.items() if k.startswith(prefix)}


class EpisodicMemory:
    """Chronological log of episodes with timestamps that can be persisted."""
    def __init__(self, max_episodes: int | None = None) -> None:
        self.episodes: List[Dict[str, Any]] = []
                self.max_episodes = max_episodes
              

    def log(self, entry: Dict[str, Any]) -> None:
        """Record an entry with a UTC timestamp."""
             
        entry['timestamp'] = str(datetime.datetime.utcnow())
        self.episodes.append(entry)
                if self.max_episodes is not None and len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def recall_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last n logged episodes."""
        return self.episodes[-n:]

    def save(self, path: str) -> None:
        """Persist episodes to disk via pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.episodes, f)

    def load(self, path: str) -> None:
        """Load episodes from disk."""
        with open(path, 'rb') as f:
            self.episodes = pickle.load(f)

class SemanticMemory:
    """Long-term storage mapping facts to their values."""
    def __init__(self) -> None:
        self.facts: Dict[str, Any] = {}

    def add_fact(self, key: str, value: Any) -> None:
        """Add or update a fact."""
        self.facts[key] = value

    def get_fact(self, key: str) -> Any:
        """Retrieve a fact by key, or None if absent."""
        return self.facts.get(key)
