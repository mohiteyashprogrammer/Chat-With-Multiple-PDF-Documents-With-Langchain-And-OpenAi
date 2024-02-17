from PyPDF2 import PdfReader
import streamlit  as st
import langchain
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from src.logger import logging
from src.exception import CustomException
from prompts_template.template import PROMPT_TEMPLATE
import dotenv 
import warnings
warnings.filterwarnings("ignore")

# Load the environment variables from the .env file
dotenv.load_dotenv()

# Access the environment variables just like you would with os.environ
key = os.getenv("OPENAI_API_KEY")


def get_pdf_text(pdf_docs):
    # Logging the initiation of the function
    logging.info("get pdf function called")
    try:

        # Initializing an empty string to store text extracted from PDFs
        text = ""
        # Iterating through each PDF file provided
        for pdf in pdf_docs:
            # Creating a PdfReader object for the current PDF file
            pdf_reader = PdfReader(pdf)
            # Iterating through each page in the PDF file
            for page in pdf_reader.pages:
                # Extracting text from the current page and appending it to the 'text' variable
                text += page.extract_text()
        return text
    
    except Exception as e:
        logging.info("Error Occured while getting pdf function")
        raise CustomException(e,sys)
    

def get_text_chunks(text):
    # Logging the initiation of the function
    logging.info("get_text_chunks function called")
    try:
        # Logging the initiation of the function
        logging.info("get pdf function called")

        # Creating an instance of RecursiveCharacterTextSplitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =10000, # Size of each chunk
            chunk_overlap =1000 # Overlap between consecutive chunks
        )
        # Splitting the input text into chunks using the RecursiveCharacterTextSplitter instance
        chunks = text_splitter.split_text(text)
        return chunks
    
    except Exception as e:
        logging.info("Error Occured With RecursiveCharacterTextSplitter")
        raise CustomException(e,sys)
    

def get_vector_store(text_chunks):
    """
    Creates a vector store from processed text chunks using FAISS and OpenAIEmbeddings.

    Args:
        processed_text_chunks (list): A list of pre-processed text chunks.

    Returns:
        None

    Raises:
        CustomException: If errors occur during vector store creation or saving.
    """

    logging.info("get_vector_store function called")

    try:
        # Ensure dependencies are installed
        if not isinstance(OpenAIEmbeddings(), OpenAIEmbeddings):
            raise ImportError("OpenAIEmbeddings is not installed or not accessible.")
        if not isinstance(FAISS(), FAISS):
            raise ImportError("FAISS is not installed or not accessible.")

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        # Save the vector store (consider alternative formats if needed)
        vector_store.save_local("faiss_index")

    except Exception as e:
        # Handle errors more informatively
        logging.error(f"Error occurred in get_vector_store: {e}")
        raise CustomException(e, sys)

    logging.info("Vector store created and saved successfully.")
        
def get_conversional_chain():
    try:

        # Load prompt template from file or configuration
        prompt_template = PROMPT_TEMPLATE

        # Initialize ChatOpenAI with API key and model name
        model = ChatOpenAI(
            openai_api_key = key,
            model_name = "gpt-4-turbo-preview",
            temperature = 0.3
        )
        # Create PromptTemplate object with specific variables
        prompt = PromptTemplate(
            template = prompt_template,
            input_variables=["context","question"]
        )
        # Generate conversational chain using the model and prompt
        chain = load_qa_chain(
            model,
            chain_type="stuff",
            prompt=prompt
        )
        # Return the generated chain
        return chain
    
    except Exception as e:
            logging.info("Error Occured With get_conversional_chain")
            raise CustomException(e,sys)
    

def user_input(user_question):
    """
    Processes a user's question and generates a response using a conversational chain.

    Args:
        user_question (str): The question asked by the user.

    Returns:
        None
    """
    try:
        # Create a utility for generating text embeddings
        embeddings = OpenAIEmbeddings()
        # Load a FAISS index for fast similarity search
        new_db = FAISS.load_local("faiss_index",embeddings)
        # Find documents similar to the user's question
        docs = new_db.similarity_search(user_question)
        # Get a pre-generated conversational chain
        chain = get_conversional_chain()

        response = chain( # Call the chain to generate a response
            {"input_documents":docs,"question":user_question}, # Provide context documents and question
            return_only_outputs=True # Request only the output text
        )
        print(response)# Print the full response object (for debugging)
        st.write("Reply: ", response["output_text"]) # Display the generated text to the user

    except Exception as e:
            logging.info("Error Occured With user_input")
            raise CustomException(e,sys)
    