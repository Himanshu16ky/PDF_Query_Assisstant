# 📄 PDF Query Assistant

An intelligent, conversational AI application that allows you to chat with your PDF documents. Upload a file, and get accurate, context-aware answers from a variety of powerful Large Language Models (LLMs).

### 🚀 Application Demo
click on the image below to see demo.

[![Alt text for your video thumbnail](https://drive.google.com/uc?export=view&id=1IEbgthI9g28FNzFW5QNoijdqubqueUra)](https://drive.google.com/file/d/1LdniHb6cXIgnLZdFOa8f1oGlkF0srYwc/view?usp=sharing)


---

## ✨ Features

* **Interactive Chat Interface:** Ask questions about your PDF in a natural, conversational way.
* **Drag-and-Drop PDF Upload:** Easily upload any PDF document directly into the application.
* **Multi-Model Support:** Choose from a selection of state-of-the-art language models to generate your answers:
    * Google's **Gemini-Flash**
    * Groq's **Llama-3.3**
    * **Cohere**'s Command R+
* **Adjustable Context Retrieval:** Use a slider to control the number of relevant text chunks (`k`) sent to the model, allowing you to balance between context depth and response speed.
* **View Sources:** Each answer is accompanied by the exact source text from the PDF that was used to generate it, ensuring transparency and trust.
* **Persistent Chat History:** Your conversation is saved for the current session, allowing you to ask follow-up questions.
* **Easy Controls:** Clear the chat history or upload a new document with the click of a button.

---

## 🛠️ Tech Stack & Architecture

This application is built with a modern, efficient tech stack designed for AI-powered applications.

* **Frontend:** [Streamlit](https://streamlit.io/) - For the interactive web UI.
* **Backend & Orchestration:** [LangChain](https://www.langchain.com/) - To connect and chain all the different components of the RAG pipeline.
* **LLM APIs:**
    * [Google Gemini API](https://ai.google.dev/)
    * [Groq API](https://groq.com/)
    * [Cohere API](https://cohere.com/)
* **PDF Parsing:** [pdfplumber](https://github.com/jsvine/pdfplumber) - For robust text extraction from PDF files.
* **Embedding Model:** `all-MiniLM-L6-v2` (via `sentence-transformers`) - A powerful, local model for creating text embeddings.
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) - For efficient local storage and retrieval of text embeddings.

### How It Works (RAG Architecture)

1.  **Upload & Parse:** The user uploads a PDF. The application uses `pdfplumber` to extract all text content.
2.  **Chunking:** The extracted text is split into smaller, overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
3.  **Embedding:** Each text chunk is converted into a numerical vector (an embedding) using the `all-MiniLM-L6-v2` model.
4.  **Storage:** These embeddings are stored locally in a ChromaDB vector store, creating a searchable knowledge base for the document.
5.  **Query & Retrieval:** When a user asks a question:
    a. The question is also converted into an embedding.
    b. ChromaDB performs a similarity search to find the most relevant text chunks from the PDF.
6.  **Generation:** The user's question, along with the retrieved text chunks, is sent to the selected LLM (Gemini, Groq, or Cohere). The LLM then generates a final, context-aware answer.

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.9+
* An IDE like VS Code

### 1. Clone the Repository

```bash
git clone https://github.com/Himanshu16ky/PDF_Query_Assisstant.git
cd PDF_Query_Assisstant
```

### 2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# For macOS/Linux
```
python3 -m venv venv
source venv/bin/
```

# For Windows
```
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
Create a requirements.txt file with the following content:

streamlit
langchain
langchain-google-genai
langchain-groq
langchain-cohere
python-dotenv
pdfplumber
sentence-transformers
chromadb

Then, install all the required packages:
```
pip install -r requirements.txt
```

### 4. Configure API Keys
You will need API keys for the services you want to use.
Create a file named .env in the root of your project directory.
Add your API keys to this file. The application will load them automatically. The APIs I used were all free of cost at the time of development.

# .env file
```
GOOGLE_API_KEY = <your-google-api-key>
GROQ_API_KEY = <your-groq-api-key>
COHERE_API_KEY = <your-cohere-api-key>
```

### 5. Run the Application
Once everything is set up, launch the Streamlit app with the following command:
```
streamlit run app.py
```
Your web browser should automatically open to the application's UI.

# 📖 How to Use
Upload a PDF: Use the file uploader in the sidebar to select a PDF document from your computer.

Wait for Processing: The application will process the document and create the vector store. A success message will appear when it's ready.

Configure Model: In the sidebar, select your desired AI model from the dropdown and adjust the "Number of Chunks" slider if needed.

Ask a Question: Type your question into the chat input at the bottom of the screen and press Enter.

View Answer & Sources: The AI's response will appear in the chat window. You can click on the "View Sources" expander to see the exact text from the PDF that informed the answer.
