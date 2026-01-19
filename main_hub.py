import streamlit as st

st.set_page_config(page_title="Text Agents by Mazahar", layout="wide")

# Define the pages in the sidebar
pages = [
    st.Page("agents/classifier.py", title="Thematic Classifier", icon="🏷️"),
    st.Page("agents/analyzer.py", title="Pattern Analyzer", icon="🧩")
]

pg = st.navigation(pages)

# Sidebar Header Branding
st.sidebar.markdown("# 🏢 Enterprise AI Agents by Mazahar")
st.sidebar.info("Select an agent to begin.")

pg.run()
