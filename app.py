import streamlit as st
import PyPDF2 as PDFReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MODEL_NAME = "gpt-3.5-turbo"

def get_pdf_text(pdf_docs):
    """Extract text from PDF files with error handling"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PDFReader.PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks with overlap for context preservation"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create vector store from text chunks"""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(text_chunks, embeddings)

def get_conversation_chain(vector_store):
    """Create conversation chain with memory"""
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.3  # Lower temperature for more factual responses
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        verbose=True
    )

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.header("Chat with your Documents 💬")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question and st.session_state.conversation:
        # Process question
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message.type == "human":
                    with st.chat_message("user"):
                        st.write(message.content)
                else:
                    with st.chat_message("assistant"):
                        st.write(message.content)

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here and click Process",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload PDF files first!")
                return

            with st.status("Processing documents..."):
                # Extract text
                st.write("Extracting text from PDFs...")
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text:
                    st.error("Failed to extract text from PDFs")
                    return

                # Split text
                st.write("Splitting text into chunks...")
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                st.write("Creating vector database...")
                vector_store = get_vector_store(text_chunks)
                
                # Create conversation chain
                st.write("Initializing chatbot...")
                st.session_state.conversation = get_conversation_chain(vector_store)
                
            st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    main()