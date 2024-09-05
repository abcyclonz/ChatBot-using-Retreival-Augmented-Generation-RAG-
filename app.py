import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

MODEL = "llama2"
model = Ollama(model=MODEL)
parser = StrOutputParser()

# Initialize the HuggingFace model and tokenizer
hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModel.from_pretrained(hf_model_name)

# Function to get embeddings using HuggingFace model
def get_hf_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = hf_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Load and split the PDF
loader = PyPDFLoader('sample1.pdf')
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

# Extract text from Document objects
chunk_texts = [chunk.page_content for chunk in chunks]
chunk_embeddings = get_hf_embeddings(chunk_texts)

# Create Faiss index
d = chunk_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(chunk_embeddings)

# Function to retrieve documents using Faiss
def retrieve_faiss(query, index, chunks):
    query_embedding = get_hf_embeddings([query])
    D, I = index.search(query_embedding, k=1)
    return I[0], [chunks[i].page_content for i in I[0]]

# Prompt template
template = """
You are an expert assistant. Given the context provided below, your task is to answer the user's question in a concise, clear, and accurate manner. 
You must ONLY use the information in the context to answer the question. If the answer is not found in the context, explicitly state: "Can't answer, Question is out of context."

Follow these instructions:
- Analyze the user's question carefully.
- Identify the key information from the context that answers the question.
- If multiple pieces of information are relevant, combine them logically to formulate your answer.
- Avoid guessing or providing information that is not explicitly present in the context.

====================
CONTEXT:
{context}

====================
USER QUESTION:
{question}

====================
ANSWER:
"""
prompt = PromptTemplate.from_template(template)

# Function to generate response
def generate_response(question: str) -> str:
    indices, context = retrieve_faiss(question, index, chunks)
    context_text = "\n".join(context)
    print(context_text)
    prompt_text = prompt.format(context=context_text, question=question)
    print(prompt_text)
    
    response = model(prompt_text)
    print(response)
    
    parsed_response = parser.parse(response)

    return parsed_response

# Print vector store
def print_vector_store():
    print("Vector Store Embeddings:")
    for i, embedding in enumerate(chunk_embeddings):
        print(f"Document {i}: {embedding}")

# Print chunks
def print_chunks():
    print("Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.page_content}")

# Print retrieved data
def print_retrieved_data(query):
    indices, retrieved_docs = retrieve_faiss(query, index, chunks)
    print(f"Retrieved data for query '{query}':")
    for i, doc in zip(indices, retrieved_docs):
        print(f"Chunk {i}: {doc}")

# UI setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="ChatBot", page_icon="A")
st.title("Hi")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Type..")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = generate_response(user_query)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(response))

    # Print debug information
    # print_vector_store()
    # print_chunks()
    print_retrieved_data(user_query)

# Tests
def test_embeddings():
    example_text = ["This is a test document."]
    example_embeddings = get_hf_embeddings(example_text)
    assert example_embeddings is not None, "Embeddings should not be None"
    assert len(example_embeddings) == 1, "Embeddings should have one entry"
    assert len(example_embeddings[0]) > 0, "Embedding vector should not be empty"
    print("Embedding test passed!")

def test_retriever():
    query = "Sample query"
    results = retrieve_faiss(query, index, chunks)
    assert results is not None, "Results should not be None"
    assert len(results) > 0, "Results should contain at least one entry"
    print("Retriever test passed!")

# Run tests
test_embeddings()
test_retriever()
