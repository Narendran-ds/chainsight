"""
prompt_templates.py — ChainSight Narration Layer, prompt construction
Pure, stateless helpers that turn a RuleEvent dict (or a list of them) into
a Gemini prompt string. No API calls, no state — mirrors rules/rule_definitions.py's
role for the rule engine, so prompt wording can be unit-tested independently
of gemini_client.py.

Design principle: the narration layer is constrained rephrasing, not a second
reasoning stage. Every prompt below repeats the same instruction — describe
only what the deterministic rule engine already decided, no causal claims, no
blame, no severity/risk judgments beyond what's in the input — so the LLM
narrator can't quietly become a second learned decision-maker in the pipeline
(see docs/scope_and_limitations.md §3).
"""

from typing import Dict, List

SYSTEM_INSTRUCTION = (
    "You are rephrasing structured safety-rule outputs into plain English for "
    "a human reviewer. Do not add causal claims, blame, severity judgments, or "
    "safety conclusions beyond what is explicitly stated in the input data. "
    "Stick to describing what was observed."
)


def build_event_prompt(event: Dict) -> str:
    """
    One RuleEvent dict (rule_id, rule_name, frame, timestamp, severity,
    track_ids, trigger, evidence, conclusion) -> a prompt asking for a single
    plain-English sentence describing it.
    """
    return (
        f"Rule fired: {event.get('rule_name')} ({event.get('rule_id')})\n"
        f"Frame: {event.get('frame')} (t={event.get('timestamp')}s)\n"
        f"Track IDs involved: {event.get('track_ids')}\n"
        f"Trigger: {event.get('trigger')}\n"
        f"Evidence: {event.get('evidence')}\n"
        f"Conclusion: {event.get('conclusion')}\n\n"
        "Rewrite the above as a single, plain-English sentence for a human "
        "safety reviewer. Report only what the trigger/evidence/conclusion "
        "already state — do not infer intent, fault, or risk level."
    )


def build_summary_prompt(events: List[Dict]) -> str:
    """
    All fired RuleEvent dicts for one clip -> a prompt asking for one
    aggregate paragraph describing counts and patterns across the run.
    """
    counts: Dict[str, int] = {}
    for e in events:
        name = e.get("rule_name", e.get("rule_id", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    counts_lines = "\n".join(f"- {name}: {n}" for name, n in counts.items())

    event_lines = "\n".join(
        f"- frame {e.get('frame')}: {e.get('rule_name')} — {e.get('conclusion')}"
        for e in events
    )

    return (
        f"A clip was analyzed by a deterministic warehouse-safety rule engine. "
        f"{len(events)} rule event(s) fired in total.\n\n"
        f"Counts by rule:\n{counts_lines}\n\n"
        f"Individual events:\n{event_lines}\n\n"
        "Write one short paragraph summarizing these counts and patterns for "
        "a human reviewer. Describe only what was counted and observed above "
        "— do not add causal claims, risk-culture judgments, or conclusions "
        "about negligence or intent that aren't directly supported by the "
        "listed events."
    )
