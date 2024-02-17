import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import get_conversional_chain,get_pdf_text,get_text_chunks,get_vector_store,user_input

import streamlit as st


def main():
    """
    Main function handles user interaction and processes user questions.
    """
    logging.info("Main function is Executing...")
    try:
        # Set page configuration and header
        st.set_page_config("Chat PDFs")
        st.header("Ask Questions To PDFs")

        # Prompt user for question
        user_question = st.text_input("Ask Questions To PDFs")

        # Process user question if provided
        if user_question:
             user_input(user_question)

        with st.slider:
             st.title("Chat PDFs")

             # Upload multiple PDF files
             pdf_docs = st.file_uploader(
                  "Upload your PDF Files and Click on the Submit & Process Button",
                  accept_multiple_files=True
             )
             # Process files upon button click
             if st.button("Submit & Process"):
                  with st.spinner("Processing..."):
                       # Extract text from PDFs
                       raw_text = get_pdf_text(pdf_docs)

                        # Split text into manageable chunks
                       text_chunks = get_text_chunks(raw_text)

                        # Generate vector representation of text chunks
                       get_vector_store(text_chunks)
                       
                       st.success("Done!")

    except Exception as e:
            logging.info("Error Occured With main function")
            raise CustomException(e,sys)