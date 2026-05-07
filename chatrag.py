import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # For displaying results
import os

# Load embedding model (must match the one used to create DB)
# @st.cache_resource  # Cache for faster reloads
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

PERSIST_DIR = os.path.join(os.path.dirname(__file__), r"C:\Users\kanna\Desktop\GUVI\5.Project\Final Project\chroma_db")  # ← absolute path

# PERSIST_DIR = r"C:\Users\kanna\Desktop\GUVI\5.Project\Final Project\chroma_db"
# @st.cache_resource  # Cache the DB load
def load_chroma_db(PERSIST_DIR):
    embedding_model = load_embedding_model()
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    return db




# Streamlit UI
st.title("Chroma DB Semantic Search App")
st.write("Enter a query to find similar documents from your vector store.")
st.write("Loading from:", PERSIST_DIR)
st.write("Exists:", os.path.exists(PERSIST_DIR))

# Load DB once
try:
    db = load_chroma_db(PERSIST_DIR)
    st.success(f"Loaded DB with {db._collection.count()} documents.")
except Exception as e:
    st.error(f"Failed to load DB: {e}")
    st.stop()

# User input
query = st.text_input("Enter your search query:", placeholder="e.g., 'What is AI?'")

if query:
    # Perform similarity search (top 5 results)
    results = db.similarity_search(query, k=5)
    
    st.subheader(f"Top {len(results)} Similar Documents:")
    
    # Display results
    for i, doc in enumerate(results, 1):
        with st.expander(f"Result {i}"): 
                        
            st.write("**Content Preview:**")
            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            # st.write("**Metadata:**", doc.metadata)

# {doc.metadata.get('source', 'Unknown')[:50]}..."):

# Optional: Add retriever for chains (e.g., with LLM)
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=your_llm, retriever=db.as_retriever())
# response = qa_chain.run(query)
# st.write("LLM Answer:", response)

