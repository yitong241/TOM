# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastAPI import generate_text

class QuestionRequest(BaseModel):
    question: Optional[str] = None
    system_prompt_description: Optional[str] = None

app = FastAPI()

@app.post("/generate-answer")
async def generate_answer(request_data: QuestionRequest):
    question = request_data.question
    system_prompt_description = request_data.system_prompt_description or "You are a friendly chatbot."
    
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    answer = generate_text(question, system_prompt_description)
    
    if answer == "No question provided.":
        raise HTTPException(status_code=400, detail=answer)
    
    return {"answer": answer}
