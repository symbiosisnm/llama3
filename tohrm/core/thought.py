# core/thought.py

from typing import List, Dict, Optional, Any
import uuid
import json

class Thought:
    def __init__(self, 
                 content: str, 
                 step_type: str = "plan", 
                 parent_id: Optional[str] = None, 
                 confidence: float = 1.0, 
                 state: str = "pending", 
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.step_type = step_type      # e.g., plan, hypothesis, action, reflection
        self.parent_id = parent_id
        self.confidence = confidence
        self.state = state              # e.g., pending, executed, failed, reflected
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "step_type": self.step_type,
            "parent_id": self.parent_id,
            "confidence": self.confidence,
            "state": self.state,
            "metadata": self.metadata
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)

    def update_state(self, new_state: str, metadata_update: Optional[Dict[str, Any]] = None) -> None:
        """Update the state of this thought and optionally merge new metadata.

        Args:
            new_state: The new state string (e.g., 'executed', 'failed').
            metadata_update: Optional dict of metadata to merge with existing metadata.
        """
        self.state = new_state
        if metadata_update:
            self.metadata.update(metadata_update)

class ThoughtBuffer:
    def __init__(self):
        self.thoughts: List[Thought] = []

    def add(self, thought: Thought):
        self.thoughts.append(thought)

    def prune(self, condition):
        self.thoughts = [t for t in self.thoughts if not condition(t)]

    def get_active(self):
        return [t for t in self.thoughts if t.state == "pending"]

    def get_by_type(self, step_type: str):
        return [t for t in self.thoughts if t.step_type == step_type]

    def get_last(self, n=1):
        return self.thoughts[-n:]

    def to_list(self):
        return [t.to_dict() for t in self.thoughts]

    def __repr__(self):
        return json.dumps(self.to_list(), indent=2)
