import streamlit as st
from pathlib import Path

_LOGO = Path(__file__).resolve().parent / 'images' / 'nba_logo.svg'

_LOGO_CSS = '''
<style>
[data-testid='stSidebarHeader'] {
    padding: 2rem 1.2rem 1.2rem;
    overflow: visible !important;
}
[data-testid='stSidebarHeader'] img {
    height: 80px !important;
    width: auto !important;
    margin-top: 0.5rem;
}
</style>
'''

def setup_logo():
    if _LOGO.exists():
        st.logo(str(_LOGO), size='large')
    st.markdown(_LOGO_CSS, unsafe_allow_html=True)
