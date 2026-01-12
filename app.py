import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import time

# --- SECURE SETUP ---
st.set_page_config(page_title="Idenifying Data Patterns by Mazahar", layout="wide")

# Ensure your secret is named exactly GOOGLE_API_KEY in Streamlit Settings > Secrets
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")

st.title("🎯 Universal Thematic Agent by Mazahar")

# 1. DEFINE BUCKETS
st.header("1. Define Your Custom Categories")
num_buckets = st.number_input("How many themes?", 1, 10, 3)
user_buckets = {}
cols = st.columns(num_buckets)
for i in range(num_buckets):
    with cols[i]:
        b_name = st.text_input(f"Name {i+1}", f"Theme {i+1}", key=f"n{i}")
        b_desc = st.text_area(f"Definition {i+1}", key=f"d{i}", placeholder="Paste messy text here...")
        user_buckets[b_name] = b_desc

# 2. DATA UPLOAD
st.header("2. Upload & Map Data")
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.write(f"Loaded {len(df):,} rows.")
    selected_cols = st.multiselect("Select text columns to analyze", df.columns)
    
    if st.button("🚀 Start Gemini 3 Analysis") and selected_cols:
        df['combined_context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
        results = []
        progress_bar = st.progress(0)
        
        # BATCHING: 30 rows per request stays under RPM limits
        batch_size = 30
        for i in range(0, len(df), batch_size):
            batch = df['combined_context'].iloc[i:i+batch_size].tolist()
            
            prompt = f"""
            TASK: Categorize these texts into: {list(user_buckets.keys())}
            DEFINITIONS (JSON): {json.dumps(user_buckets)}
            RULES: Return ONLY the exact bucket name for each, one per line.
            TEXTS: {batch}
            """
            
            try:
                # Using the modern Gemini 3 client syntax
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level="low")
                    )
                )
                
                if response.text:
                    labels = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
                    results.extend(labels[:len(batch)])
                    while len(results) < (i + len(batch)):
                        results.append("Uncategorized")
                else:
                    results.extend(["Uncategorized"] * len(batch))
                    
            except Exception as e:
                results.extend(["API_Error"] * len(batch))
            
            progress_bar.progress(min((i + batch_size) / len(df), 1.0))
            time.sleep(5) # Base delay for Free Tier stability

        df['AI_Result'] = results[:len(df)]
        st.success("Analysis Complete!")
        st.bar_chart(df['AI_Result'].value_counts())
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")
