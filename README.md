# Codex-RAG

A modern **Multimodal Retrieval-Augmented Generation (RAG)** assistant that enables users to interact with PDF documents and images using Mistral AI, LangChain, and ChromaDB.

The application builds a local vector database from uploaded PDFs, retrieves relevant context, and generates grounded responses. It also supports multimodal understanding by incorporating image inputs into the conversation.

---

## Features

- PDF-based Retrieval-Augmented Generation (RAG)
- Image understanding with Mistral Vision
- ChromaDB vector database
- Semantic document search
- Grounded responses using retrieved context
- Modern Streamlit interface
- Conversation history
- Source chunk retrieval
- Local knowledge base creation

---

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Mistral AI
- Mistral Vision
- FAISS / Vector Embeddings
- PyPDF
- Pillow

---

## Project Structure

```text
Codex-RAG/
│
├── multimodal_app.py
├── create_database.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/codex-rag.git
cd codex-rag
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate it

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
MISTRAL_API_KEY=YOUR_API_KEY
```

---

## Run the application

```bash
streamlit run multimodal_app.py
```

---

## How It Works

1. Upload a PDF document.
2. Build the vector database.
3. Upload an optional image.
4. Ask questions about the uploaded content.
5. The application retrieves relevant document chunks.
6. Mistral AI generates a grounded response using the retrieved context.

---

## Key Capabilities

- Retrieval-Augmented Generation
- Semantic Search
- Multimodal Question Answering
- Document Understanding
- Image Understanding
- Context-Aware Responses
- Local Vector Database
- Interactive Chat Interface

---

## Future Improvements

- Multiple PDF support
- Citation highlighting
- Conversation export
- OCR support
- User authentication
- Cloud deployment
- Streaming responses

---

## Screenshots

### Home

> Add screenshot here

### Chat Interface

> Add screenshot here

### Document Retrieval

> Add screenshot here

---

## Author

**Pranav Garg**

GitHub: https://github.com/supremeinferno

LinkedIn: *Add your LinkedIn profile*

---

## License

This project is licensed under the MIT License.