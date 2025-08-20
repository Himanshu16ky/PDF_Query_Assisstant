import os
import pdfplumber
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere 
load_dotenv()

def process_and_create_vector_store(uploaded_file, temp_dir="temp_pdf_data"):
    """
    Processes an uploaded PDF file and creates a fresh vector store.
    It saves the file temporarily to read it.
    """
    # Create a temporary directory to store the uploaded file
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Define a unique directory for this PDF's database (for FAISS)
    pdf_basename = os.path.splitext(uploaded_file.name)[0]
    db_directory = os.path.join(temp_dir, f"{pdf_basename}_faiss_db")

    print(f"Processing {uploaded_file.name} and creating a new vector store...")

    # --- Process the PDF using pdfplumber ---
    all_docs = []
    with pdfplumber.open(temp_file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={'source': uploaded_file.name, 'page': i + 1})
                all_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)

    # --- Create and save the new FAISS vector store ---
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Check if FAISS index already exists for this PDF
    if os.path.exists(db_directory):
        try:
            # Load existing FAISS index
            db = FAISS.load_local(
                db_directory, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS vector store.")
        except Exception as e:
            print(f"Error loading existing index: {e}. Creating new one...")
            # Create new FAISS index if loading fails
            db = FAISS.from_documents(texts, embeddings)
            # Save the FAISS index
            db.save_local(db_directory)
            print("New FAISS vector store created and saved.")
    else:
        # Create new FAISS index
        db = FAISS.from_documents(texts, embeddings)
        # Save the FAISS index
        db.save_local(db_directory)
        print("New FAISS vector store created and saved.")

    return db

def get_qa_chain(db, model_name="Gemini-Flash", k=2):
    """
    Creates and returns a RetrievalQA chain configured with the specified model and number of chunks (k).
    """
    print(f"Setting up QA chain with model: {model_name} and k={k}")

    # --- Model Selection Logic ---
    if model_name == "Gemini-Flash":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    elif model_name == "Grok llama-3.3":
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.5
        )
    elif model_name == "Cohere":
        llm = ChatCohere()
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True
    )
    return qa_chain

# Optional: Utility functions for manual FAISS operations
def save_faiss_index(db, file_path):
    """Save FAISS index to specified path"""
    db.save_local(file_path)
    print(f"FAISS index saved to {file_path}")

def load_faiss_index(file_path, embeddings):
    """Load FAISS index from specified path"""
    try:
        db = FAISS.load_local(
            file_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"FAISS index loaded from {file_path}")
        return db
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None
