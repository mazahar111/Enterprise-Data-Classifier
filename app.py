import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import time
import io

# --- SECURE API SETUP ---
# Update this in Streamlit Dashboard > Settings > Secrets: GOOGLE_API_KEY = "AIza..."
if "GOOGLE_API_KEY" in st.secrets:
    # Use the new Google Gen AI SDK for Gemini 3
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing API Key! Please paste it into the Streamlit Secrets vault.")

st.set_page_config(page_title="Gemini 3 Enterprise Agent", layout="wide")
st.title("🚀 Gemini 3 Flash: Thematic Classifier")

# --- 1. DYNAMIC BUCKETS ---
st.header("1. Define Your Custom Categories")
st.info("Tip: You can use symbols like $, (), and quotes in your definitions.")
num_buckets = st.number_input("How many categories?", 1, 10, 3)

user_buckets = {}
cols = st.columns(num_buckets)
for i in range(num_buckets):
    with cols[i]:
        b_name = st.text_input(f"Name {i+1}", f"Theme {i+1}", key=f"n{i}")
        b_desc = st.text_area(f"Definition {i+1}", key=f"d{i}", placeholder="Paste messy text...")
        user_buckets[b_name] = b_desc

# --- 2. MULTI-FORMAT UPLOAD ---
st.header("2. Upload & Map Data")
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.write(f"Loaded {len(df):,} rows.")
    selected_cols = st.multiselect("Select text columns to analyze", df.columns)
    
    if st.button("🚀 Run Gemini 3 Analysis") and selected_cols:
        df['context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
        results = []
        progress_bar = st.progress(0)
        
        # BATCHING: 30 rows per request
        batch_size = 30
        for i in range(0, len(df), batch_size):
            batch = df['context'].iloc[i:i+batch_size].tolist()
            
            prompt = f"""
            TASK: Categorize these texts into: {list(user_buckets.keys())}
            DEFINITIONS (JSON): {json.dumps(user_buckets)}
            Return ONLY the bucket name for each, one per line.
            TEXTS: {batch}
            """
            
            # Retry Loop for Rate Limits
            success = False
            retries = 0
            while not success and retries < 3:
                try:
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL")
                        )
                    )
                    labels = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
                    results.extend(labels[:len(batch)])
                    # Pad if AI skipped any
                    while len(results) < (i + len(batch)):
                        results.append("Uncategorized")
                    success = True
                except Exception as e:
                    retries += 1
                    st.warning(f"Rate limit hit. Waiting 60s to retry batch {i//batch_size + 1}...")
                    time.sleep(60) # Wait for quota reset
            
            progress_bar.progress(min((i + batch_size) / len(df), 1.0))
            time.sleep(5) # Base delay for Free Tier

        df['AI_Result'] = results[:len(df)]
        st.success("Analysis Complete!")
        st.bar_chart(df['AI_Result'].value_counts())
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")
