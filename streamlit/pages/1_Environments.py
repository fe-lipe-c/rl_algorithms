import pandas as pd
import numpy as np
import streamlit as st
from base.render_setup import render_sidebar, page_style

page_style()


def run_interface():
    page_title = "ENVIRONMENTS"
    st.markdown(
        rf"""
        <div style="color: {st.session_state.text_color}; font-size: 1.1rem; text-align: justify; margin-top: 20px; padding: 50px;">
            <h4 style="color: #c7cc45;">{page_title}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


def init_session():
    pass


def main():
    init_session()
    render_sidebar()
    run_interface()


if __name__ == "__main__":
    main()
