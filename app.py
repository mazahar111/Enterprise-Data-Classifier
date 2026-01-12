import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time

# --- SECURE API SETUP ---
# The code looks for the key you just pasted in the "Secrets" box
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing API Key! Please paste it into the Streamlit Secrets vault.")

st.title("🎯 Enterprise Dynamic Classifier by Maz")
st.markdown("Analyzes up to 200k rows across multiple columns.")

# --- 1. DYNAMIC BUCKETS (HANDLES SPECIAL CHARACTERS) ---
st.header("1. Define Your Themes")
num_buckets = st.number_input("How many categories?", 1, 10, 3)

user_buckets = {}
cols = st.columns(num_buckets)
for i in range(num_buckets):
    with cols[i]:
        b_name = st.text_input(f"Name {i+1}", f"Theme {i+1}", key=f"n{i}")
        # Definitions can include special characters or messy copy-pastes
        b_desc = st.text_area(f"Definition {i+1}", key=f"d{i}")
        user_buckets[b_name] = b_desc

# --- 2. DYNAMIC FILE UPLOAD ---
st.header("2. Upload & Map Data")
file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file)
    st.write(f"Loaded {len(df):,} rows.")
    selected_cols = st.multiselect("Select columns to analyze", df.columns)
    
    if st.button("🚀 Start Analysis") and selected_cols:
        # Pre-processing: Combine columns and handle empty cells
        df['context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        results = []
        progress = st.progress(0)
        
        # Batching: Process 50 rows at a time to stay within rate limits
        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df['context'].iloc[i:i+batch_size].tolist()
            
            # Robust prompt using JSON to safely pass special characters
            prompt = f"""
            Task: Categorize each text entry into one of these buckets: {list(user_buckets.keys())}
            Definitions (JSON): {json.dumps(user_buckets)}
            
            Return ONLY the bucket name for each entry, one per line.
            Texts: {batch}
            """
            
            try:
                response = model.generate_content(prompt)
                # Clean AI output and handle potential formatting noise
                labels = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
                results.extend(labels[:len(batch)])
                
                # Fallback if AI provides fewer answers than rows
                while len(results) < (i + len(batch)):
                    results.append("Uncategorized")
                    
            except Exception as e:
                results.extend(["Error"] * len(batch))
            
            progress.progress(min((i + batch_size) / len(df), 1.0))
            time.sleep(1) # Rate limit safety

        df['AI_Theme'] = results[:len(df)]
        st.success("Analysis Complete!")
        st.bar_chart(df['AI_Theme'].value_counts())
        st.download_button("Download CSV", df.to_csv(index=False), "analyzed_data.csv")
