"""
internals.py — ChainSight demo, raw pipeline internals
Collapsed-by-default expander showing each stage's raw JSON, for anyone who
wants to dig past the rules/narration headline view (see CLAUDE.md's Demo
layer section — this is deliberately secondary, not the main view).
"""

import streamlit as st


def render(tracks: dict, spatial_events: dict, world_graph_summary: dict) -> None:
    with st.expander("Pipeline internals (raw JSON)", expanded=False):
        st.markdown("**tracks.json**")
        st.json(tracks, expanded=False)
        st.markdown("**spatial_events.json**")
        st.json(spatial_events or {}, expanded=False)
        st.markdown("**world_graph_summary.json**")
        st.json(world_graph_summary or {}, expanded=False)
