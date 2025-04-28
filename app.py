import sys
import torch
sys.modules["torch.classes"] = torch.classes = type("classes", (), {"__path__": []})()

import streamlit as st
from src.retriever import Retriever
from src.generator import Generator

# 🎨 Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f5f7fa; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3 { color: #0F52BA; }
        .stButton>button {
            background-color: #0F52BA;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)




st.title("🩺 MedicalGPT")
st.markdown("Enter a clinical query to retrieve relevant patient cases and generate diagnostic insights.")

# 📥 Input
query = st.text_input("🔍 Query")

# 📌 Sidebar
st.sidebar.title("ℹ️ Instructions")
st.sidebar.markdown("""
- Enter a clinical question based on patient symptoms, history, or labs.
- The system retrieves relevant patient cases.
- It uses an AI model to generate a diagnosis or reasoning.
""")

# 🚀 Trigger RAG
if query:
    with st.spinner("🔍 Retrieving relevant clinical cases..."):
        retriever = Retriever()
        docs = retriever.search(query)
        contexts = docs["text"].tolist()

    st.markdown("### 📄 Top Retrieved Clinical Notes:")
    for i, ctx in enumerate(contexts):
        with st.expander(f"Context {i+1}"):
            st.code(ctx[:1000], language="text")

    with st.spinner("💡 Generating diagnostic insight..."):
        generator = Generator()
        answer = generator.generate(query, contexts)

    st.markdown("### ✅ AI-Suggested Diagnosis:")
    st.success(answer)
