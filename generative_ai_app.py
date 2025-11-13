import streamlit as st
import os
import uuid
import tempfile
from datetime import datetime
import json
from typing import List, Dict, Any

# LangChain imports
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

# Try to import the newer huggingface embeddings first
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Additional imports
import chromadb
from chromadb.config import Settings

class GenerativeAIApp:
    """
    A comprehensive Generative AI application with RAG capabilities,
    session management, and PDF processing using Groq API.
    """
    
    def __init__(self):
        """Initialize the application with default settings."""
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_embeddings()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Generative AI Assistant with RAG",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            
        if 'groq_api_key' not in st.session_state:
            st.session_state.groq_api_key = None
            
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
            
        if 'rag_chain' not in st.session_state:
            st.session_state.rag_chain = None
            
        if 'pdf_processed' not in st.session_state:
            st.session_state.pdf_processed = False
            
    def setup_embeddings(self):
        """Setup HuggingFace embeddings for vector operations."""
        try:
            # Use the updated import and configuration for better compatibility
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': True
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
        except ImportError:
            # Fallback to community embeddings if langchain-huggingface is not available
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                st.error(f"Error setting up embeddings: {str(e)}")
                self.embeddings = None
        except Exception as e:
            st.error(f"Error setting up embeddings: {str(e)}")
            self.embeddings = None
            
    def get_groq_api_key(self):
        """Get Groq API key from user input."""
        st.sidebar.header("🔑 API Configuration")
        
        api_key = st.sidebar.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Get your API key from https://console.groq.com/keys"
        )
        
        if api_key:
            st.session_state.groq_api_key = api_key
            st.sidebar.success("✅ API Key configured!")
            return True
        else:
            st.sidebar.warning("⚠️ Please enter your Groq API Key to continue")
            return False
            
    def initialize_groq_llm(self):
        """Initialize Groq LLM with the provided API key."""
        if not st.session_state.groq_api_key:
            return None
            
        try:
            llm = ChatGroq(
                groq_api_key=st.session_state.groq_api_key,
                model_name="Gemma2-9b-it",
                temperature=0.1,
                max_tokens=1024
            )
            return llm
        except Exception as e:
            st.error(f"Error initializing Groq LLM: {str(e)}")
            return None
            
    def process_pdf(self, uploaded_file):
        """Process uploaded PDF file and create vector store."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=f"./chroma_db_{st.session_state.session_id}"
            )
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            return vector_store, len(splits)
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None, 0
            
    def create_rag_chain(self, vector_store):
        """Create RAG chain for conversational retrieval."""
        try:
            llm = self.initialize_groq_llm()
            if not llm:
                return None
                
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 5},
                search_type="similarity"
            )
            
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                max_tokens_limit=2000
            )
            rag_chain.return_source_documents = True
            
            return rag_chain
            
        except Exception as e:
            st.error(f"Error creating RAG chain: {str(e)}")
            return None
            
    def display_session_info(self):
        """Display current session information."""
        st.sidebar.header("📊 Session Information")
        st.sidebar.write(f"**Session ID:** `{st.session_state.session_id}`")
        st.sidebar.write(f"**Messages:** {len(st.session_state.conversation_history)}")
        st.sidebar.write(f"**PDF Processed:** {'✅' if st.session_state.pdf_processed else '❌'}")
        
        if st.sidebar.button("🔄 New Session"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.conversation_history = []
            st.session_state.vector_store = None
            st.session_state.rag_chain = None
            st.session_state.pdf_processed = False
            st.rerun()
            
    def display_conversation_history(self):
        """Display conversation history."""
        if st.session_state.conversation_history:
            st.subheader("💬 Conversation History")
            for i, message in enumerate(st.session_state.conversation_history):
                with st.expander(f"Message {i+1} - {message['timestamp']}"):
                    st.write(f"**User:** {message['user']}")
                    st.write(f"**Assistant:** {message['assistant']}")
                    
    def get_conversation_context(self):
        """Get formatted conversation history for context."""
        if not st.session_state.conversation_history:
            return ""
        
        context = "\n\nPrevious conversation history:\n"
        for i, msg in enumerate(st.session_state.conversation_history, 1):
            context += f"Q{i}: {msg['user']}\n"
            context += f"A{i}: {msg['assistant']}\n\n"
        
        return context

    def handle_user_query(self, query: str):
        """Handle user query with RAG or direct LLM response."""
        if not st.session_state.groq_api_key:
            st.error("Please configure your Groq API key first!")
            return
            
        # Check if user is asking about conversation history
        history_keywords = ['questions', 'asked', 'conversation', 'history', 'previous', 'earlier', 'before']
        is_history_query = any(keyword in query.lower() for keyword in history_keywords)
        
        # Get conversation context
        conversation_context = self.get_conversation_context()
        history_tuples = [
            (item['user'], item['assistant'])
            for item in st.session_state.conversation_history
        ]
        
        if st.session_state.pdf_processed and st.session_state.rag_chain:
            # Use RAG chain for PDF-based queries
            try:
                with st.spinner("🤔 Thinking..."):
                    # Enhance query with conversation context if asking about history
                    enhanced_query = query
                    if is_history_query and conversation_context:
                        enhanced_query = f"{query}\n\n{conversation_context}"
                    response = st.session_state.rag_chain.invoke({
                        "question": enhanced_query,
                        "chat_history": history_tuples
                    })
                    
                answer = response['answer']
                sources = response.get('source_documents', [])
                
                # Display answer
                st.write("**🤖 Assistant:**")
                st.write(answer)
                
                # Display sources if available
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources):
                            st.write(f"**Source {i+1}:**")
                            st.write(source.page_content[:200] + "...")
                            st.write(f"**Page:** {source.metadata.get('page', 'N/A')}")
                            st.write("---")
                
                # Store in conversation history
                st.session_state.conversation_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'user': query,
                    'assistant': answer,
                    'type': 'rag'
                })
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                
        else:
            # Use direct LLM for general queries
            try:
                llm = self.initialize_groq_llm()
                if llm:
                    with st.spinner("🤔 Thinking..."):
                        # Enhance query with conversation context
                        enhanced_query = query
                        if conversation_context:
                            enhanced_query = f"{query}\n\n{conversation_context}"
                        
                        response = llm.invoke(enhanced_query)
                    
                    answer = response.content
                    
                    # Display answer
                    st.write("**🤖 Assistant:**")
                    st.write(answer)
                    
                    # Store in conversation history
                    st.session_state.conversation_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'user': query,
                        'assistant': answer,
                        'type': 'general'
                    })
                    
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                
    def main(self):
        """Main application interface."""
        st.title("🤖 Generative AI Assistant with RAG")
        st.markdown("---")
        
        # Sidebar configuration
        self.display_session_info()
        
        # API Key configuration
        if not self.get_groq_api_key():
            st.info("👈 Please configure your Groq API key in the sidebar to get started!")
            return
            
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Chat Interface")
            
            # PDF Upload Section
            st.subheader("📄 Upload PDF for RAG")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document to enable RAG (Retrieval Augmented Generation)"
            )
            
            if uploaded_file is not None:
                if st.button("🔄 Process PDF"):
                    with st.spinner("Processing PDF..."):
                        vector_store, num_chunks = self.process_pdf(uploaded_file)
                        
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.session_state.pdf_processed = True
                            
                            # Create RAG chain
                            rag_chain = self.create_rag_chain(vector_store)
                            if rag_chain:
                                st.session_state.rag_chain = rag_chain
                                st.success(f"✅ PDF processed successfully! Created {num_chunks} chunks.")
                            else:
                                st.error("❌ Failed to create RAG chain.")
                        else:
                            st.error("❌ Failed to process PDF.")
                            
            # Chat Interface
            st.subheader("💭 Ask Questions")
            
            # Show conversation history prompts if there's history
            if st.session_state.conversation_history:
                st.info("💡 **Tip:** You can ask me about our previous conversation! Try asking: 'What questions have I asked you so far?' or 'Summarize our conversation'")
            
            # Text input for user query
            user_input = st.text_area(
                "Enter your question:",
                height=100,
                placeholder="Ask me anything! If you've uploaded a PDF, I can answer questions about it using RAG. I also remember our entire conversation history!"
            )
            
            if st.button("🚀 Send Message", type="primary"):
                if user_input.strip():
                    self.handle_user_query(user_input.strip())
                else:
                    st.warning("Please enter a question!")
                    
        with col2:
            st.header("📋 Quick Actions")
            
            # Quick conversation history queries
            if st.session_state.conversation_history:
                st.subheader("💬 Conversation Queries")
                if st.button("❓ What questions have I asked?"):
                    self.handle_user_query("What questions have I asked you so far in our conversation?")
                if st.button("📝 Summarize our conversation"):
                    self.handle_user_query("Can you summarize our entire conversation so far?")
                if st.button("🔍 What topics did we discuss?"):
                    self.handle_user_query("What topics and subjects have we discussed in our conversation?")
            
            # Display conversation history
            self.display_conversation_history()
            
            # Export conversation
            if st.session_state.conversation_history:
                if st.button("📥 Export Conversation"):
                    conversation_data = {
                        'session_id': st.session_state.session_id,
                        'timestamp': datetime.now().isoformat(),
                        'conversation': st.session_state.conversation_history
                    }
                    
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(conversation_data, indent=2),
                        file_name=f"conversation_{st.session_state.session_id}.json",
                        mime="application/json"
                    )
                    
            # Clear conversation
            if st.button("🗑️ Clear Conversation"):
                st.session_state.conversation_history = []
                st.rerun()
                
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
                <p>🤖 Generative AI Assistant | Powered by Groq API | Built with Streamlit & LangChain</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function to run the application."""
    app = GenerativeAIApp()
    app.main()

if __name__ == "__main__":
    main()
