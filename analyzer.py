import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import time

# --- SECURE SETUP ---
if "GOOGLE_API_KEY" in st.secrets:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing GOOGLE_API_KEY in Secrets!")

st.title("🧩 Advanced Pattern Analyzer")
st.markdown("Select specific columns to identify high-level archetypes and standard patterns.")

# 1. DATA UPLOAD
file = st.file_uploader("Upload Report", type=["csv", "xlsx"], key="ana_up")

if file:
    # Read file based on extension
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    st.write(f"Loaded {len(df):,} rows.")

    # 2. DYNAMIC COLUMN SELECTION
    # We pre-fill the selection with text-based columns as a suggestion
    default_cols = df.select_dtypes(include=['object']).columns.tolist()
    selected_cols = st.multiselect(
        "Which columns should be analyzed for patterns?", 
        options=df.columns.tolist(),
        default=default_cols
    )
    
    if st.button("🔍 Extract Standard Patterns") and selected_cols:
        # Create a combined context string from the user's selected columns
        df['mega_context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
        
        results = []
        progress_bar = st.progress(0)
        
        # BATCHING: 30 rows per request to stay under free tier limits
        batch_size = 30
        for i in range(0, len(df), batch_size):
            batch = df['mega_context'].iloc[i:i+batch_size].tolist()
            
            prompt = f"""
            TASK: Identify the high-level 'Standard Archetype' for these entries.
            RULES:
            - Use a professional 3-word theme (e.g., 'Systemic Workflow Delay').
            - Focus on the underlying cause/pattern.
            - Return ONLY the list of themes, one per line.
            ENTRIES:
            {batch}
            """
            
            try:
                # Using Gemini 3 Flash Preview for advanced reasoning
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
                else:
                    results.extend(["Pattern Unknown"] * len(batch))
                    
            except Exception as e:
                st.warning(f"Batch {i//batch_size + 1} failed: {str(e)}")
                results.extend(["API_Error"] * len(batch))
            
            # Progress and mandatory delay for Free Tier stability
            progress_bar.progress(min((i + batch_size) / len(df), 1.0))
            time.sleep(5) 

        df['Standard_Pattern'] = results[:len(df)]
        st.success("Analysis Complete!")
        
        # 3. RESULTS VISUALIZATION
        st.subheader("📊 Pattern Distribution")
        st.bar_chart(df['Standard_Pattern'].value_counts())
        
        # Show only selected columns and the new result for clarity
        st.dataframe(df[selected_cols + ['Standard_Pattern']])
        
        # Universal CSV Download
        st.download_button("Download Results", df.to_csv(index=False), "pattern_analysis.csv")
