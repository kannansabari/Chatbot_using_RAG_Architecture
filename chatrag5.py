# Without heatmap and relevant score > 50

import streamlit as st
import streamlit.components.v1 as components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
import ollama
import os

# Config
PERSIST_DIR = r"C:\Users\kanna\Desktop\GUVI\5.Project\Final Project\chroma_db"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Loaders
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

def load_chroma_db(persist_dir):
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=load_embedding_model()
    )

def load_reranker():
    return CrossEncoder(RERANKER_MODEL)

# Streamlit UI 
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("🔍 RAG Chat with Source Viewer")
st.caption(f"DB path: `{PERSIST_DIR}` | Exists: {os.path.exists(PERSIST_DIR)}")

# Load resources
try:
    db       = load_chroma_db(PERSIST_DIR)
    reranker = load_reranker()
    st.success(f"✅ Loaded DB — {db._collection.count()} documents.")
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Main Chat Loop 
if prompt := st.chat_input("Ask a question..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1. Retrieve with scores
    results_with_scores = db.similarity_search_with_score(prompt, k=10)

    # 2. Rerank using cross-encoder
    docs   = [doc for doc, _ in results_with_scores]
    pairs  = [[prompt, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in ranked[:3]]

    # 3. Display sources panel
    st.markdown("---")
    st.subheader("📄 Retrieved Sources")

    # Filter only sources with relevance > 50%
    relevant_sources = [
        (score, doc) for score, doc in ranked 
        if round((score + 10) / 20 * 100, 1) > 50
    ][:3]

    if not relevant_sources:
        st.info("No sources met the minimum relevance threshold of 50%.")
        exit()
    else:
        cols = st.columns(len(relevant_sources))
        
        for i, (score, doc) in enumerate(relevant_sources):
            # Calculate percentage and convert to native Python float
            relevance_pct = round(float(score + 10) / 20 * 100, 1)
            relevance_pct = max(0.0, min(100.0, relevance_pct))   # Safety clamp
            
            with cols[i]:
                st.markdown(f"**Source {i+1}**")
                
                # ✅ Fixed: Convert to native float
                st.progress(float(relevance_pct / 100))
                
                st.caption(f"Relevance: `{relevance_pct}%`")
                
                with st.expander("View chunk"):
                    st.write(doc.page_content)

    # 4. Build context and call LLM
    context = "\n\n".join([doc.page_content for doc in top_docs])
    ollama_messages = [
        {"role": "system", "content": f"Use the context below to answer:\n\n{context}"},
        *st.session_state.messages
    ]

    with st.chat_message("assistant"):
        response = ollama.chat(model="tinyllama", messages=ollama_messages)
        reply = response.message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})

