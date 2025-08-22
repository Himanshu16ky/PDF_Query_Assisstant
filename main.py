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
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_and_create_vector_store(uploaded_file, temp_dir="temp_pdf_data"):
    """
    Processes an uploaded PDF file and creates a fresh vector store.
    It saves the file temporarily to read it.
    """
    try:
        # Validate input file
        if uploaded_file is None:
            raise ValueError("No file uploaded")
        
        if uploaded_file.size == 0:
            raise ValueError("Uploaded file is empty")
        
        # Create a temporary directory to store the uploaded file
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write file with error handling
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"File saved to {temp_file_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save uploaded file: {str(e)}")

        # Define a unique directory for this PDF's database (for FAISS)
        pdf_basename = os.path.splitext(uploaded_file.name)[0]
        db_directory = os.path.join(temp_dir, f"{pdf_basename}_faiss_db")

        logger.info(f"Processing {uploaded_file.name} and creating a new vector store...")

        # --- Process the PDF using pdfplumber ---
        all_docs = []
        try:
            with pdfplumber.open(temp_file_path) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF contains no pages")
                
                logger.info(f"PDF has {len(pdf.pages)} pages")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():  # Check for non-empty text
                            doc = Document(
                                page_content=text.strip(), 
                                metadata={'source': uploaded_file.name, 'page': i + 1}
                            )
                            all_docs.append(doc)
                            logger.info(f"Extracted text from page {i + 1}: {len(text)} characters")
                        else:
                            logger.warning(f"Page {i + 1} contains no extractable text")
                    except Exception as e:
                        logger.error(f"Error extracting text from page {i + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logger.info("Temporary file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

        # Check if we have any documents
        if not all_docs:
            raise ValueError("No readable text found in the PDF. The PDF might be image-based, corrupted, or password-protected.")
        
        logger.info(f"Created {len(all_docs)} documents from PDF")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        try:
            texts = text_splitter.split_documents(all_docs)
        except Exception as e:
            raise RuntimeError(f"Failed to split documents: {str(e)}")
        
        # Validate text chunks
        if not texts:
            raise ValueError("No text chunks were created from the PDF")
        
        # Filter out empty chunks
        valid_texts = [text for text in texts if text.page_content.strip()]
        
        if not valid_texts:
            raise ValueError("All text chunks are empty after processing")
        
        logger.info(f"Created {len(valid_texts)} valid text chunks")

        # --- Create and save the new FAISS vector store ---
        try:
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")
        
        # Check if FAISS index already exists for this PDF
        if os.path.exists(db_directory):
            try:
                # Load existing FAISS index
                db = FAISS.load_local(
                    db_directory, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS vector store.")
            except Exception as e:
                logger.error(f"Error loading existing index: {e}. Creating new one...")
                # Create new FAISS index if loading fails
                try:
                    db = FAISS.from_documents(valid_texts, embeddings)
                    # Save the FAISS index
                    db.save_local(db_directory)
                    logger.info("New FAISS vector store created and saved.")
                except Exception as create_error:
                    raise RuntimeError(f"Failed to create FAISS vector store: {str(create_error)}")
        else:
            # Create new FAISS index
            try:
                db = FAISS.from_documents(valid_texts, embeddings)
                # Save the FAISS index
                db.save_local(db_directory)
                logger.info("New FAISS vector store created and saved.")
            except Exception as e:
                raise RuntimeError(f"Failed to create and save FAISS vector store: {str(e)}")

        return db
        
    except Exception as e:
        logger.error(f"Error in process_and_create_vector_store: {str(e)}")
        raise

def get_qa_chain(db, model_name="Gemini-Flash", k=2):
    """
    Creates and returns a RetrievalQA chain configured with the specified model and number of chunks (k).
    """
    try:
        logger.info(f"Setting up QA chain with model: {model_name} and k={k}")
        
        # Validate inputs
        if db is None:
            raise ValueError("Vector database is None")
        
        if k <= 0:
            raise ValueError("k must be greater than 0")

        # --- Model Selection Logic ---
        if model_name == "Gemini-Flash":
            try:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")
                
        elif model_name == "Grok llama-3.3":
            try:
                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.5
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
                
        elif model_name == "Cohere":
            try:
                llm = ChatCohere()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Cohere model: {str(e)}")
        else:
            raise ValueError(f"Model '{model_name}' is not supported.")

        # Create QA chain
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": k}),
                return_source_documents=True
            )
            return qa_chain
        except Exception as e:
            raise RuntimeError(f"Failed to create QA chain: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in get_qa_chain: {str(e)}")
        raise

# Optional: Utility functions for manual FAISS operations
def save_faiss_index(db, file_path):
    """Save FAISS index to specified path"""
    try:
        if db is None:
            raise ValueError("Database is None")
        db.save_local(file_path)
        logger.info(f"FAISS index saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {str(e)}")
        raise

def load_faiss_index(file_path, embeddings):
    """Load FAISS index from specified path"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAISS index not found at {file_path}")
            
        db = FAISS.load_local(
            file_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS index loaded from {file_path}")
        return db
    except Exception as e:
        logger.error(f"Error loading FAISS index: {str(e)}")
        return None
