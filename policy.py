from typing import Dict, Optional

# ------------------------------
# Policy store
# ------------------------------
class PolicyStore:
    def __init__(self, policies: Optional[Dict[str, str]] = None):
        self.policies = policies or {}

    def add_policy(self, role: str, policy_text: str):
        self.policies[role] = policy_text

    def resolve(self, role: str, epoch: str) -> str:
        if role not in self.policies:
            raise ValueError(f"No policy registered for role '{role}'")
        return self.policies[role]

