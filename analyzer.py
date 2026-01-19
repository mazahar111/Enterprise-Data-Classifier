import streamlit as st
import pandas as pd
from google import genai
import time

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.title("🧩 Advanced Pattern Analyzer")
st.write("Automatically extracts high-level standard patterns across all text.")

file = st.file_uploader("Upload Report", type=["csv", "xlsx"], key="ana_up")
if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    
    # AUTO-DETECT: Find all columns containing text (strings/objects)
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    st.info(f"Detected {len(text_cols)} text columns for analysis.")

    if st.button("🔍 Extract Standard Patterns"):
        df['mega_context'] = df[text_cols].fillna('').agg(' | '.join, axis=1)
        results = []
        
        for i in range(0, len(df), 30):
            batch = df['mega_context'].iloc[i:i+30].tolist()
            prompt = f"Analyze these entries and provide a 3-word 'Standard Archetype' (e.g., 'Systemic Workflow Delay') for each. Return only themes: {batch}"
            
            response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
            results.extend(response.text.strip().split('\n'))
            time.sleep(5)

        df['AI_Pattern'] = results[:len(df)]
        st.success("Patterns Identified!")
        st.bar_chart(df['AI_Result'].value_counts() if 'AI_Result' in df else df['AI_Pattern'].value_counts())
        st.dataframe(df)
