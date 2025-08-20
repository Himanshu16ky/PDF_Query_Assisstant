import streamlit as st
import time
from main import process_and_create_vector_store, get_qa_chain

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# Session State
# Storing variables that persist across reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db" not in st.session_state:
    st.session_state.db = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# UI 
st.title("📄 PDF Query Assistant")
st.markdown("""
Welcome! Upload a PDF, select your preferred AI model, and ask any questions you have about the document.
""")

# Leftbar for Controls
with st.sidebar:
    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        # Display the uploaded file's name
        st.markdown(f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <p style="margin: 0;"><b>Uploaded:</b> {uploaded_file.name}</p>
        </div>
        """, unsafe_allow_html=True)

        # Process the PDF if it's new or has changed
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.processing = True
            try:
                with st.spinner("Processing PDF... This may take a moment."):
                    st.session_state.db = process_and_create_vector_store(uploaded_file)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.chat_history = [] # Clear history for new file
                    st.session_state.processing = False
            except Exception as e:
                st.session_state.processing = False
                st.error(f"Error processing PDF: {e}. Please try a different PDF.")
                st.session_state.db = None
                st.session_state.uploaded_file_name = None
            if st.session_state.db is not None:
                st.success("PDF processed successfully! You can now ask questions.")
    else:
        st.session_state.db = None
        st.session_state.uploaded_file_name = None
        st.session_state.processing = False

    # Model and Chunk Selection by user
    st.subheader("Model Configuration")
    model_name = st.selectbox(
        "Select AI Model",
        ("Gemini-Flash","Grok llama-3.3","Cohere"),
        key="model_selector"
    )

    k_chunks = st.slider(
        "Number of Chunks to Retrieve",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
        help="How many relevant text chunks from the PDF should be sent to the model? Higher values can provide more context but may increase processing time."
    )
    
    #  Clear Chat Button 
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main Chat Interface 
st.subheader("Chat with your Document")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Status indicator
if st.session_state.processing:
    st.info("🔄 Processing PDF... You can type your question, but please wait for processing to complete before submitting.")
elif st.session_state.db is None:
    st.warning("📁 Please upload a PDF file to start asking questions.")

# Chat input bar 
query = st.chat_input("Ask a question about your PDF...")

if query:
    # Check if database is ready before processing
    if st.session_state.db is None:
        st.error("⚠️ Please upload and wait for the PDF to be processed before asking questions.")
    elif st.session_state.processing:
        st.error("⚠️ PDF is still being processed. Please wait a moment and try again.")
    else:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Generating most relavant answer..."):
                # Get the QA chain and run the query
                qa_chain = get_qa_chain(st.session_state.db, model_name, k_chunks)
                try:
                    result = qa_chain({"query": query})
                    response_text = result["result"]
                    for chunk in response_text.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    # Display source documents
                    with st.expander("View Sources"):
                        for doc in result["source_documents"]:
                            st.info(f"Source: {doc.metadata['source']} - Page: {doc.metadata['page']}")
                            st.code(doc.page_content)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                except Exception as e:
                        st.error(f"❗ Error: {e} occurred while processing your query.\nPlease try again with a different Model.")
                        full_response = "I apologize, but I encountered an error while processing your query. Please try again with a different model or try again later."
                        message_placeholder.markdown(full_response)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    
                
