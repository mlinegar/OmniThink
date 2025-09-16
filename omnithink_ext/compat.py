from typing import Tuple, Set

class EdgeCompat:
    """
    Enforces (tau(parent), rho, tau(child)) ∈ 𝓐 loaded from configs/compat_schema.yaml.
    See docs/MATH.md (Edge compatibility).
    """
    def __init__(self, schema):
        self.A: Set[Tuple[str, str, str]] = set(map(tuple, schema["compatibility"]))

    def check(self, parent_tau: str, rho: str, child_tau: str) -> bool:
        return (parent_tau, rho, child_tau) in self.A
