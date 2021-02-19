import streamlit as st

import os
from utils import (
    fetch_news_for_summarization, 
    get_pegasus_for_summarization,
    get_summaries
)

UPPER_LIMIT = 5

if __name__ == '__main__':

    model, tokenizer = get_pegasus_for_summarization(model_id="human-centered-summarization/financial-summarization-pegasus")    

    st.write("""
    # Stock Prediction
    ## Summary
    """)

    fetch_news_button = st.button("fetch news")
    if fetch_news_button:
        if "articles_for_summarization.txt" in os.listdir("metadata/"):
            with open("metadata/articles_for_summarization.txt", "r") as f:
                articles = f.read().split("\n")
            summaries = get_summaries(model, tokenizer, articles[:UPPER_LIMIT])
            st.markdown("""
            # Financial News
            """)
            for s in summaries:
                st.write(s)
        else:
            st.write("Sorry! articles are not available")
