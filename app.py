import streamlit as st
import pandas as pd
import google.generativeai as genai
import re

st.set_page_config(page_title="Enterprise AI Agent", layout="wide")

# Secure API Connection
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.sidebar.warning("API Key not found in Secrets. Use sidebar for testing.")
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if api_key: genai.configure(api_key=api_key)

st.title("🏢 Enterprise Universal Theme Classifier")
st.markdown("---")

# 1. DYNAMIC BUCKETS & CLEANING
st.header("1. Define Your Custom Categories")
num_buckets = st.number_input("How many themes do you need?", 1, 10, 3)

user_buckets = {}
cols = st.columns(num_buckets)
for i in range(num_buckets):
    with cols[i]:
        b_name = st.text_input(f"Theme {i+1} Name", f"Theme {i+1}")
        b_desc = st.text_area(f"Paste Definition {i+1}", "Copy/Paste messy text here...")
        # Cleaning the definition on the fly
        cleaned_desc = re.sub(r'\s+', ' ', b_desc).strip()
        user_buckets[b_name] = cleaned_desc

# 2. DYNAMIC DATA LOADING
st.header("2. Upload Your Dataset")
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.write(f"Loaded {len(df):,} rows.")
    
    selected_cols = st.multiselect("Which columns should the AI read?", df.columns)
    
    if st.button("🚀 Run Analysis") and selected_cols:
        # Pre-processing Data (Cleansing duplicates and nulls)
        df = df.dropna(subset=selected_cols).drop_duplicates(subset=selected_cols)
        df['context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        results = []
        progress_bar = st.progress(0)
        
        # Batch Processing for Scale (200k rows)
        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df['context'].iloc[i:i+batch_size].tolist()
            prompt = f"Categorize into: {list(user_buckets.keys())}. Rules: {user_buckets}. Return ONLY the name per line: {batch}"
            
            try:
                response = model.generate_content(prompt)
                results.extend([l.strip() for l in response.text.strip().split('\n') if l.strip()])
            except:
                results.extend(["Error"] * len(batch))
            progress_bar.progress(min((i + batch_size) / len(df), 1.0))

        df['AI_Theme'] = results[:len(df)]
        st.success("Analysis Complete!")
        st.bar_chart(df['AI_Theme'].value_counts())
        st.download_button("Download Analyzed CSV", df.to_csv(index=False), "results.csv")
