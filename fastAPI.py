from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Define the model ID for the transformer model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize the pipeline for text generation
# Set device to 0 if CUDA is available, else -1 (for CPU)
device = 0 if torch.cuda.is_available() else -1
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

class QuestionRequest(BaseModel):
    question: Optional[str] = None

app = FastAPI()

@app.post("/generate-answer")
async def generate_answer(request_data: QuestionRequest):
    question = request_data.question
    
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    # Format the prompt with the question
    prompt = f"Question: {question}\n\nAnswer: "
    
    try:
        # Generate an answer using the transformers pipeline
        response = text_generator(prompt, max_length=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        # Extract the generated text and remove the input prompt
        answer = response[0]['generated_text'][len(prompt):]
        return {"answer": answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
