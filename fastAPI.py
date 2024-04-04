from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import pipeline

# Define the FastAPI app
app = FastAPI()

# Initialize the pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# Pydantic model for request validation
class TextGenerationRequest(BaseModel):
    messages: Optional[List[str]] = None

@app.post("/generate-text")
async def generate_text(request_data: TextGenerationRequest):
    messages = request_data.messages
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]
        
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn fastAPI:app --reload
