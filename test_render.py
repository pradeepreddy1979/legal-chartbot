import streamlit as st
import os

st.set_page_config(page_title="Test App", layout="wide")

st.markdown("""
<style>
/* Remove all background colors to ensure nothing is hiding content */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    background-color: white !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.title("✅ Streamlit Rendering Check")
st.sidebar.title("✅ Sidebar Check")

# Check if the .env file is accessible
if os.path.exists(".env"):
    st.success("Configuration file (.env) found.")
else:
    st.error("Configuration file (.env) MISSING.")