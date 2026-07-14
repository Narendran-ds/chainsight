"""
narrator.py — ChainSight Narration Layer, orchestration
Consumes rules/engine.py's rule_events.json and produces plain-English
narration: one sentence per fired RuleEvent, plus one aggregate summary
paragraph for the whole clip.

Pipeline position:
    rules/engine.py (rule_events.json) -> narration/narrator.py (THIS)
        -> outputs/narration_<run>.json -> Streamlit demo

Design principle: narration is constrained rephrasing of an already-fired,
deterministic RuleEvent — never a new judgment (see prompt_templates.py's
SYSTEM_INSTRUCTION). Written as a separate outputs/narration_<run>.json
rather than mutating rule_events.json in place, so existing consumers of
rule_events.json's schema (tests, pipeline.py) are unaffected by adding this
stage.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from . import prompt_templates
from .gemini_client import GeminiClient

logger = logging.getLogger("chainsight.narration.narrator")

NO_EVENTS_SUMMARY = "No rule events were fired in this clip."


class Narrator:
    """Turns fired RuleEvents into human-readable narration via GeminiClient."""

    def __init__(self, client: GeminiClient):
        self.client = client

    # --- loading (14) ---
    def load_rule_events(self, rule_events_path: str) -> List[Dict]:
        path = Path(rule_events_path)
        if not path.exists():
            raise FileNotFoundError(f"rule_events.json not found: {rule_events_path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"rule_events.json is not valid JSON: {e}") from e
        if not isinstance(data, list):
            raise ValueError(
                f"rule_events.json schema mismatch: expected a list of rule "
                f"events, got {type(data).__name__}."
            )
        return data

    # --- narration ---
    def narrate_events(self, events: List[Dict]) -> List[Dict]:
        """Returns copies of each event dict with a "narration" key added."""
        narrated = []
        for event in events:
            prompt = prompt_templates.build_event_prompt(event)
            narration = self.client.generate(prompt, prompt_templates.SYSTEM_INSTRUCTION)
            narrated.append({**event, "narration": narration})
        return narrated

    def summarize(self, events: List[Dict]) -> str:
        if not events:
            return NO_EVENTS_SUMMARY
        prompt = prompt_templates.build_summary_prompt(events)
        return self.client.generate(prompt, prompt_templates.SYSTEM_INSTRUCTION)

    def run(self, rule_events_path: str) -> Dict:
        events = self.load_rule_events(rule_events_path)
        narrated_events = self.narrate_events(events)
        summary = self.summarize(events)
        logger.info(f"Narrated {len(narrated_events)} event(s); summary generated.")
        return {"events": narrated_events, "summary": summary}
