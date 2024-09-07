from fastapi import FastAPI, File, UploadFile, HTTPException
import requests
from bs4 import BeautifulSoup
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel

app = FastAPI()

# Initialize the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory storage
storage = {}

# Define a base model to accept URL input as a JSON object
class URLInput(BaseModel):
    url: str

class ChatRequest(BaseModel):
    chat_id: str
    question: str

# Define a root route
@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Endpoint to process a web URL
@app.post("/process_url/")
async def process_web_url(input_data: URLInput):
    try:
        # Extracts URL from JSON object
        url = input_data.url

        # Scrape content from the URL
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=' ', strip=True)
        
        # Generate a unique chat ID (simplified here as URL)
        chat_id = url
        
        # Store the cleaned content
        storage[chat_id] = text
        
        return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to process a PDF document
@app.post("/process_pdf/")
async def process_pdf(file: UploadFile = File(...)):
    try:
        # Extract text from the uploaded PDF
        text = ''
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        
        # Generate a unique chat ID (simplified here as file name)
        chat_id = file.filename
        
        # Store the cleaned content
        storage[chat_id] = text
        
        return {"chat_id": chat_id, "message": "PDF content processed and stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat API to query the stored content
@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        chat_id= request.chat_id
        question= request.question

        # Retrieve the stored content
        content = storage.get(chat_id)
        if not content:
            raise HTTPException(status_code=404, detail="Chat ID not found.")
        
        # Split content into sentences for better similarity matching
        sentences = content.split('.')  # Simple sentence splitting, can be improved
        content_embeddings = model.encode(sentences, convert_to_tensor=True)
        question_embedding = model.encode(question, convert_to_tensor=True)
        
        # Calculate similarity and find the most relevant response
        similarity = util.pytorch_cos_sim(question_embedding, content_embeddings)
        most_similar_index = similarity.argmax()
        most_similar_sentence = sentences[most_similar_index]
        
        return {"response": most_similar_sentence}
    except Exception as e:
        print("Error in /chat/ endpoint:", str(e)) #log the error
        raise HTTPException(status_code=500, detail=str(e))