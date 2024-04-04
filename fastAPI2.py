from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import torch
from transformers import pipeline

class Message(BaseModel):
    role: str
    content: str

class TextGenerationRequest(BaseModel):
    messages: Optional[List[Message]] = None

def extract_first_system_response(generated_text):
    parts = generated_text.split(" user:")
    first_response = parts[1] 
    return first_response


app = FastAPI()
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")

@app.post("/generate-text")
async def generate_text(request_data: TextGenerationRequest):
    messages = request_data.messages
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    try:
        prompt = " ".join([f"{msg.role}: {msg.content}" for msg in messages])
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]
        print(generated_text)
        first_system_response = extract_first_system_response(generated_text)
        return {"generated_text": first_system_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
