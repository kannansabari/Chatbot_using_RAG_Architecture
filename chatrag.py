import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document  # For displaying results

# Load embedding model (must match the one used to create DB)
@st.cache_resource  # Cache for faster reloads
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

@st.cache_resource  # Cache the DB load
def load_chroma_db(persist_dir="./chroma_db"):
    embedding_model = load_embedding_model()
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return db

# Streamlit UI
st.title("Chroma DB Semantic Search App")
st.write("Enter a query to find similar documents from your vector store.")

# Load DB once
try:
    db = load_chroma_db()
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
        with st.expander(f"Result {i}: {doc.metadata.get('source', 'Unknown')[:50]}..."):
            st.write("**Content Preview:**")
            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            st.write("**Metadata:**", doc.metadata)

# Optional: Add retriever for chains (e.g., with LLM)
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=your_llm, retriever=db.as_retriever())
# response = qa_chain.run(query)
# st.write("LLM Answer:", response)






# import streamlit as st
# import ollama

# st.title("Chat with Ollama")
# st.markdown("A simple chatbot powered by *LLM* running locally.")
# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     if msg['role'] == 'user':
#         st.chat_message("user").markdown(msg["content"])
#     else:
#         st.chat_message("assistant").markdown(msg["content"])

# if prompt := st.chat_input("Type your message..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").markdown(prompt)

#     with st.chat_message('assistant'):
#         response = ollama.chat(model = "llama2", messages=st.session_state.messages)
#         reply = response["message"]["content"]
#         st.markdown(reply)
#     st.session_state.messages.append({"role": "assistant", "content": reply})

#     query = "Explain self attention"
# retrival_docs = db.similarity_search(query, k=2)
# for doc in retrival_docs:
#     print (doc.page_content)
#     print ("\n")
#     print ("\n")

#     # Follow up question

# query = "Explain self attention"
# retrival_docs = db.similarity_search(query, k=2)
# first_doc = retrival_docs[0].page_content

# follow_up_question = r"Explain attention heads based on the content" + first_doc
# follow_up_result = db.similarity_search(follow_up_question)
# if follow_up_result[0]:
#     print(follow_up_result[0].page_content)
# else:
#     print("No information present")

