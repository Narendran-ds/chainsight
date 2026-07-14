"""
report_card.py — ChainSight demo, narration panel
Displays outputs/narration_<run>.json's clip-level summary plus per-event
narration, highlighting whichever event matches the currently-selected
frame (st.session_state["current_frame"]). Narration is optional per run
(--narrate is opt-in on scripts/run_pipeline.py) — if narration_<run>.json
doesn't exist, shows the exact command to generate it instead of an error.
"""

import streamlit as st


def render(narration: dict, run_name: str) -> None:
    st.subheader("Narration")

    if narration is None:
        run_suffix = f"_{run_name}" if run_name else ""
        st.info(
            f"Narration not generated for this run. Generate it with:\n\n"
            f"```\npython scripts/run_narration.py "
            f"--rule-events outputs/rule_events{run_suffix}.json "
            f"--out outputs/narration{run_suffix}.json\n```"
        )
        return

    st.markdown(f"**Summary:** {narration.get('summary', '')}")

    events = narration.get("events", [])
    if not events:
        return

    current_frame = st.session_state.get("current_frame")
    st.markdown("**Per-event narration:**")
    for event in events:
        prefix = "▶️ " if event.get("frame") == current_frame else ""
        st.markdown(f"{prefix}Frame {event.get('frame')}: {event.get('narration', '')}")
