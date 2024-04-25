from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = 0 if torch.cuda.is_available() else -1
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

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
    
    # Format the prompt with the system description and the question
    prompt = f"{system_prompt_description}\nQuestion: {question}\n\nAnswer: "
    
    try:
        # Generate an answer using the transformers pipeline
        response = text_generator(prompt, max_length=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, truncation=True)
        # Extract the generated text and remove the input prompt
        answer = response[0]['generated_text'][len(prompt):]
        return {"answer": answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))