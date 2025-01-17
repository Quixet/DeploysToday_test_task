from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.chat import chat_response

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Chat API! Use /chat with POST method to send queries."}

@app.post("/chat")
async def chat(request: QueryRequest):
    response = chat_response(request.query)
    print(response)
    return {"response"}


