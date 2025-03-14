import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini API Key is missing. Please set it in the .env file.")

genai.configure(api_key=gemini_api_key)

# FastAPI App Initialization
app = FastAPI(title="Tripzoori AI Assistant API", description="API for answering questions about Tripzoori", version="1.0")

# Website URL
TRIPZOORI_URL = "https://tripzoori-gittest1.fly.dev/"

# Function to Load and Process Website Content with Selenium
def load_website_content(url):
    # Setup Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        time.sleep(5)  # Wait for JavaScript to load content

        content = driver.page_source  # Get fully rendered HTML
        driver.quit()

        document = Document(page_content=content, metadata={"source": url})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        document_chunks = text_splitter.split_documents([document])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(document_chunks, embeddings)

        return vector_store

    except Exception as e:
        driver.quit()
        raise Exception(f"Error fetching website content: {str(e)}")

# Load Vector Store
vector_store = load_website_content(TRIPZOORI_URL)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
    You are Tripzoori AI Assistant, designed to answer questions about the website 'tripzoori-gittest1.fly.dev'.
    Use the following context to provide accurate responses:

    {context}

    Question: {input}
    """
)

# Function to Generate Responses using Gemini
def generate_response(final_prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(final_prompt)
    return response.text

# Function to Create RAG Chain
def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever()
    retrieval_chain = create_stuff_documents_chain(generate_response, prompt)
    return create_retrieval_chain(retriever, retrieval_chain)

retrieval_chain = get_conversation_chain(vector_store)

# Request Model for API Input
class QueryRequest(BaseModel):
    question: str

# API Endpoint
@app.post("/ask")
def ask_tripzoori(request: QueryRequest):
    try:
        retrieved_docs = vector_store.similarity_search(request.question, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        final_prompt = prompt.format(context=context, input=request.question)
        response = generate_response(final_prompt)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
