import bs4
import requests
import streamlit as st
from tqdm import tqdm

import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

@st.cache(allow_output_mutation=True)
def get_pegasus_for_summarization(model_id:str):
    tokenizer = PegasusTokenizer.from_pretrained(model_id)
    model = PegasusForConditionalGeneration.from_pretrained(model_id)
    return model, tokenizer

@st.cache(allow_output_mutation=True)
def fetch_news_for_summarization():

    main_website = "https://www.cnbc.com/finance/"
    page = requests.get(main_website)
    soup = bs4.BeautifulSoup(page.content, 'html.parser')

    links = soup.find_all("a", class_="Card-title")
    links = [c["href"] for c in links]

    info = []

    for url in tqdm(links):
        page = requests.get(url)
        soup = bs4.BeautifulSoup(page.content, "html.parser")

        content = soup.find_all("div", class_="group")
        content = [c.text for c in content]
        info.extend(content)

    articles = "\n".join(info)
    with open("articles_for_summarization.txt", "w") as f:
        f.write(articles)
        print("ARTICLES SAVED")

    return info

def get_summaries(model, tokenizer, articles):
    input_ids = tokenizer(articles, padding=True, max_length=512, truncation=True).input_ids
    input_ids = torch.tensor(input_ids).long()
    output = model.generate(
                        input_ids,
                        max_length=32, 
                        num_beams=5,
                        early_stopping=True
                    )
    summaries = [tokenizer.decode(o, skip_special_tokens=True) for o in output]
    return summaries

        