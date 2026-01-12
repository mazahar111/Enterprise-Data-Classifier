import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import json

# ... (Previous API setup code remains the same) ...

if st.button("🚀 Run Enterprise Analysis") and selected_cols:
    # 1. CLEANING THE INPUT DATA
    df['context'] = df[selected_cols].fillna('').agg(' | '.join, axis=1)
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    results = []
    progress_bar = st.progress(0)
    
    # BATCHING: 50 rows per API call
    batch_size = 50
    for i in range(0, len(df), batch_size):
        batch = df['context'].iloc[i:i+batch_size].tolist()
        
        # WE ENCLOSE DEFINITIONS IN A CLEAR BLOCK TO HANDLE SPECIAL CHARACTERS
        prompt = f"""
        TASK: Categorize the following text list into EXACTLY one of the allowed categories.
        
        ALLOWED CATEGORIES:
        {list(user_buckets.keys())}
        
        DETAILED DEFINITIONS (Ignore formatting/special characters):
        {json.dumps(user_buckets, indent=2)}
        
        OUTPUT RULES:
        - Return ONLY the category name for each entry.
        - One category per line.
        - If unsure, pick the closest match.
        
        TEXT ENTRIES TO CATEGORIZE:
        {batch}
        """
        
        try:
            response = model.generate_content(prompt)
            if response.text:
                # Splitting by newline and cleaning whitespace/special characters from the AI output
                labels = [l.strip().replace('*', '').replace('-', '') for l in response.text.strip().split('\n') if l.strip()]
                
                # Match the AI's answers to our rows
                batch_results = labels[:len(batch)]
                
                # If AI returned too few answers, fill with the first bucket name as a fallback
                while len(batch_results) < len(batch):
                    batch_results.append("Uncategorized")
                
                results.extend(batch_results)
            else:
                results.extend(["Uncategorized"] * len(batch))
        except Exception as e:
            # This logs the specific error to your Streamlit dashboard for debugging
            st.warning(f"Batch {i} had an issue: {e}")
            results.extend(["Uncategorized"] * len(batch))
            
        # Update UI
        progress_bar.progress(min((i + batch_size) / len(df), 1.0))
        time.sleep(1) # Prevents rate-limit 'Errors' on free tier

    # Final result mapping
    df['AI_Result'] = results[:len(df)]
    st.success("Analysis Complete!")
    st.bar_chart(df['AI_Result'].value_counts())
    st.download_button("Download CSV", df.to_csv(index=False), "results.csv")
