"""
clip_picker.py — ChainSight demo, sidebar run selector
Thin Streamlit wrapper over run_data.discover_runs(). Stops the app with a
clear message if outputs/ has no runs yet, rather than rendering a broken
page with nothing to show.
"""

import streamlit as st

from . import run_data


def render(output_dir: str) -> str:
    runs = run_data.discover_runs(output_dir)
    if not runs:
        st.sidebar.error(
            f"No runs found under `{output_dir}/`. Run `scripts/run_pipeline.py` "
            f"(or the individual `scripts/run_*.py` stages) first."
        )
        st.stop()

    labels = [r if r else "(default)" for r in runs]
    selected_label = st.sidebar.selectbox("Run", labels)
    return runs[labels.index(selected_label)]
