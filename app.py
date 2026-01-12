import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time

# --- SECURE API SETUP ---
# Ensure your key is in Streamlit Dashboard > Settings > Secrets
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing API Key! Please paste it into the Streamlit Secrets vault.")

st.title("🚀 Advanced Enterprise Multi-Format Classifier by Mazahar")
st.markdown("Handles 200k+ rows with complex special characters in definitions.")

# --- 1. DYNAMIC BUCKETS (HANDLES ALL SPECIAL CHARACTERS) ---
st.header("1. Define Your Strategy")
num_buckets = st.number_input("Number of categories", 1, 10, 3)

user_buckets = {}
cols = st.columns(num_buckets)
for i in range(num_buckets):
    with cols[i]:
        # Buckets can now safely contain quotes, $, and parentheses
        b_name = st.text_input(f"Theme {i+1} Name", f"Theme {i+1}", key=f"n{i}")
        b_desc = st.text_area(f"Detailed Definition {i+1}", key=f"d{i}")
        user_buckets[b_name] = b_desc

# --- 2. MULTI-FORMAT FILE UPLOAD ---
st.header("2. Upload & Map Data")
file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if file:
    # Logic to detect file type and read accordingly
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.write(f"Loaded {len(df):,} rows.")
    selected_cols = st.multiselect("Select columns for analysis context", df.columns)
    
    if st.button("🚀 Start Advanced Analysis") and selected_cols:
        # Pre-processing: Combine columns into a single context
        df['context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        results = []
        progress = st.progress(0)
        
        # Batching: Process 30 rows at a time for stability
        batch_size = 30
        for i in range(0, len(df), batch_size):
            batch = df['context'].iloc[i:i+batch_size].tolist()
            
            # Use JSON to safely pass special characters to the AI
            prompt = f"""
            You are a strict data classifier. 
            CATEGORIES AND DEFINITIONS (JSON Format):
            {json.dumps(user_buckets, indent=2)}
            
            TASK: 
            Assign each text entry to EXACTLY one category name from the list above.
            
            RULES:
            - Return ONLY the category name.
            - One category per line.
            - Match the names exactly as they appear in the JSON keys.
            
            TEXTS TO ANALYZE:
            {batch}
            """
            
            try:
                response = model.generate_content(prompt)
                if response.text:
                    labels = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
                    # Ensure we don't exceed the batch size
                    results.extend(labels[:len(batch)])
                    
                    # Fill gaps if AI returns fewer lines than rows
                    while len(results) < (i + len(batch)):
                        results.append("Uncategorized")
                else:
                    results.extend(["Uncategorized"] * len(batch))
            except Exception as e:
                # If rate-limited (429) or other API issue
                results.extend(["API_Error"] * len(batch))
            
            # MANDATORY 2-SECOND SLEEP to stay within Free Tier limits
            progress.progress(min((i + batch_size) / len(df), 1.0))
            time.sleep(2) 

        df['AI_Result_Theme'] = results[:len(df)]
        st.success("Analysis Complete!")
        st.bar_chart(df['AI_Result_Theme'].value_counts())
        st.download_button("Download Processed CSV", df.to_csv(index=False), "results.csv")
