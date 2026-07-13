"""
events.py — ChainSight Rule Engine, event schema
Every rule produces a RuleEvent with the same shape: trigger, evidence,
conclusion. This is deliberate — a bare "unsafe: True/False" boolean
isn't explainable; a reviewer (or the narration layer later) should be
able to read a RuleEvent and understand exactly what fired and why
without re-deriving it from raw graph state.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RuleEvent:
    rule_id: str            # e.g. "R1_RESTRICTED_ZONE_INTRUSION"
    rule_name: str          # human-readable, e.g. "Restricted Zone Intrusion"
    frame: int
    timestamp: float
    severity: str           # "info" | "warning" | "critical"
    track_ids: List[int]
    trigger: str            # short statement of what fired
    evidence: dict = field(default_factory=dict)  # supporting facts (zone, distance, duration, etc.)
    conclusion: str = ""    # plain-language explanation, ready for narration/logging

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "track_ids": self.track_ids,
            "trigger": self.trigger,
            "evidence": self.evidence,
            "conclusion": self.conclusion,
        }

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.rule_name} @ frame {self.frame}: {self.conclusion}"
