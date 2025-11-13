# Generative AI Assistant with RAG

A comprehensive Streamlit application that creates an AI assistant with Retrieval Augmented Generation (RAG) capabilities using Groq API.

## Features

- 🤖 **Groq API Integration**: Uses Llama 3 8B model for fast, high-quality responses
- 📄 **PDF Processing**: Upload and process PDF documents for RAG
- 💬 **Conversation History**: Maintains context across the entire session
- 🔑 **Session Management**: Unique session IDs with persistent state
- 📊 **Vector Database**: Chroma for efficient document similarity search
- 🎯 **RAG Implementation**: Retrieval Augmented Generation for document-based Q&A
- 📥 **Export Functionality**: Export conversation history as JSON

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_app.txt
```

### 2. Get Groq API Key
1. Visit [Groq Console](https://console.groq.com/keys)
2. Create an account and generate an API key
3. Copy your API key

### 3. Run the Application
```bash
streamlit run generative_ai_app.py
```

### 4. Configure and Use
1. Enter your Groq API key in the sidebar
2. Upload a PDF file (optional) for RAG capabilities
3. Start chatting with the AI assistant!

## How It Works

1. **Session Initialization**: Creates unique session ID and initializes state
2. **API Configuration**: Secure input of Groq API key
3. **PDF Processing**: Converts PDF to chunks and creates vector embeddings
4. **RAG Chain**: Combines document retrieval with conversational AI
5. **Query Processing**: Handles both general and document-specific questions
6. **History Management**: Maintains conversation context throughout session

## File Structure

```
Cursor_AI_code/
├── generative_ai_app.py    # Main Streamlit application
├── documentation.txt       # Comprehensive code documentation
├── requirements_app.txt    # Application dependencies
└── README.md              # This file
```

## Technical Stack

- **Frontend**: Streamlit
- **AI Model**: Groq Llama 3 8B
- **Vector DB**: Chroma
- **Embeddings**: HuggingFace Sentence Transformers
- **Framework**: LangChain
- **PDF Processing**: PyPDF

## Usage Tips

- Upload PDFs for document-specific questions
- Use the conversation history to ask follow-up questions
- Export conversations for record-keeping
- Start new sessions for different topics
- The AI maintains context throughout your session

## Troubleshooting

- Ensure all dependencies are installed correctly
- Verify your Groq API key is valid
- Check that PDF files are not corrupted
- Restart the application if session state becomes corrupted

For detailed technical documentation, see `documentation.txt`.
