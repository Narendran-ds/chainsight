"""
event_timeline.py — ChainSight demo, rule-event timeline
Table of fired RuleEvents for the selected run. Clicking a row jumps the
frame scrubber (video_player.py) to that event's frame via
st.session_state["current_frame"] — this is the "front-and-center" entry
point into a clip per the demo's scoped design (see CLAUDE.md's Demo layer
section): a reviewer starts from *what fired*, not from scrubbing blind.

Uses Streamlit's native st.dataframe row-selection (added in a recent
Streamlit release) instead of the streamlit-timeline package listed in
requirements.txt — that package is a very early (0.0.2), thinly-maintained
component, and native selection is simpler and more reliable.
"""

import pandas as pd
import streamlit as st


def render(rule_events: list) -> None:
    st.subheader("Rule events")

    if not rule_events:
        st.info("No rule events fired for this run.")
        return

    table = pd.DataFrame([
        {
            "Rule": e.get("rule_name"),
            "Frame": e.get("frame"),
            "Time (s)": e.get("timestamp"),
            "Severity": e.get("severity"),
            "Trigger": e.get("trigger"),
        }
        for e in rule_events
    ])

    selection = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="event_table",
    )

    selected_rows = selection.selection.rows if selection is not None else []
    if selected_rows:
        st.session_state["current_frame"] = int(table.iloc[selected_rows[0]]["Frame"])
