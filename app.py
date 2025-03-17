import os
import tempfile
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import HfApi

# Initialize session state
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your documents."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
    if 'hf_api_key' not in st.session_state:
        st.session_state['hf_api_key'] = None

# Function to validate Hugging Face API key
def validate_huggingface_api_key(api_key):
    try:
        api = HfApi(token=api_key)
        api.model_info("google/flan-t5-small")  # Check if model is accessible
        return True
    except Exception as e:
        st.error("❌ Invalid Hugging Face API token. Please enter a valid token.")
        return False

# Conversation logic
def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

# Display chat history
def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

# Create conversational chain
def create_conversational_chain(vector_store, hf_api_key):
    retriever_llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.01, "max_length": 500},
        huggingfacehub_api_token=hf_api_key
    )

    generator_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.01, "max_length": 500},
        huggingfacehub_api_token=hf_api_key
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=generator_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain

# Main function
def main():
    initialize_session_state()
    st.title("Multi-Docs ChatBot using Two Models (Retriever + Generator)")

    # Sidebar for API Token Input
    st.sidebar.title("Settings")
    st.session_state['hf_api_key'] = st.sidebar.text_input("Enter Hugging Face API token:", type="password")

    if not st.session_state['hf_api_key']:
        st.warning("⚠️ Please enter your Hugging Face API token to proceed.")
        st.stop()

    if not validate_huggingface_api_key(st.session_state['hf_api_key']):
        st.stop()

    # Sidebar for document upload
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension in [".docx", ".doc"]:
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        # Split text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create conversational chain
        chain = create_conversational_chain(vector_store, st.session_state['hf_api_key'])

        # Display chat history
        display_chat_history(chain)

if __name__ == "__main__":
    main()
