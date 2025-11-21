import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
import numpy as np
import os

st.set_page_config(page_title="IT Helpdesk Chatbot")
st.title("💻 IT Helpdesk Chatbot")
user_query = st.text_input("Ask a question about IT issues:")

data_folder = "data"
pdf_files = ["IT HELP DESK.pdf"] 


def load_pdfs(pdf_list, folder):
    texts = []
    for file in pdf_list:
        path = os.path.join(folder, file)
        with pdfplumber.open(path) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() + "\n"
            texts.append(full_text)
    return texts

pdf_texts = load_pdfs(pdf_files, data_folder)

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = []
for text in pdf_texts:
    all_chunks.extend(split_text(text))

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(all_chunks, convert_to_numpy=True)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def search(query, k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = [all_chunks[i] for i in indices[0]]
    return results

if user_query:
    results = search(user_query, k=3)
    st.write("**Top Relevant Answers:**")
    for i, res in enumerate(results, 1):
        st.markdown(f"**{i}.** {res}")