import streamlit as st
import pandas as pd
from google import genai
import json
import time

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.title("🏷️ Thematic Classifier")
st.write("Upload a report to map rows to specific themes.")

# 1. Bucket Definitions
num_buckets = st.number_input("Number of themes", 1, 10, 3)
user_buckets = {}
cols = st.columns(num_buckets)
for i in range(num_buckets):
    with cols[i]:
        name = st.text_input(f"Theme {i+1}", f"Category {i+1}", key=f"n{i}")
        desc = st.text_area(f"Definition {i+1}", key=f"d{i}")
        user_buckets[name] = desc

# 2. Upload & Process
file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="clf_up")
if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    cols_to_map = st.multiselect("Select text columns", df.columns)
    
    if st.button("🚀 Run Classification") and cols_to_map:
        df['context'] = df[cols_to_map].fillna('').agg(' | '.join, axis=1)
        results = []
        for i in range(0, len(df), 30): # Batching for rate limits
            batch = df['context'].iloc[i:i+30].tolist()
            prompt = f"Categorize these into {list(user_buckets.keys())} using definitions: {json.dumps(user_buckets)}. Return ONLY names, one per line: {batch}"
            
            response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
            results.extend(response.text.strip().split('\n'))
            time.sleep(5) # Free tier cooldown

        df['AI_Theme'] = results[:len(df)]
        st.success("Done!")
        st.dataframe(df)
