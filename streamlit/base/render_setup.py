import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown("---")

        # style = """style='text-align: center; font-family: \"Source Code Pro\", monospace; font-size: 14px'"""
        # sidebar_markdown("<strong>Monitor de Mercado <code>v0.4</code></strong>")
        sidebar_markdown("Reinforcement Learning")
        sidebar_markdown("Felipe Costa")

        st.markdown("---")


def sidebar_markdown(text):
    """Helper function to display markdown in the sidebar"""
    style = """style='text-align: center; font-family: \"Source Code Pro\", sans-serif; font-size: 14px;'"""
    return st.markdown(f"<p {style}>{text}</p>", unsafe_allow_html=True)


def page_style():
    st.markdown(
        """<style>.block-container{max-width: 90rem !important;}</style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<style>[data-testid="stSidebar"][aria-expanded="true"]{min-width: 200px;max-width: 250px;}""",
        unsafe_allow_html=True,
    )

    # Add custom CSS for font family and size
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100;300;400;500;700;900&display=swap');
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif; /* Change to your preferred font */
            font-size: 22px; /* Base font size */
        }
        
        /* You can also target specific elements */
        h1 {
            font-size: 2rem !important;
        }
        h2 {
            font-size: 1.8rem !important;
        }
        h3 {
            font-size: 1.6rem !important;
        }
        p, div, span {
            font-size: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.text_color = st.get_option("theme.textColor")
    st.session_state.background_color = st._config.get_option("theme.backgroundColor")
