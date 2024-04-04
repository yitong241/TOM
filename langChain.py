from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
# hf = HuggingFacePipeline(pipeline=pipe)

# print(hf.generate_text("Hello, my name is", max_new_tokens=10))

template = """Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)

# chain = prompt | hf
# question = "What is electroencephalography?"
# print(chain.invoke({"question": question}))

gpu_llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    device_map="auto",  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 256, "do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
)

gpu_chain = prompt | gpu_llm

# question = "What is electroencephalography?"

# print(gpu_chain.invoke({"question": question}))

class QuestionRequest(BaseModel):
    question: Optional[str] = None

app = FastAPI()

@app.post("/generate-answer")
async def generate_answer(request_data: QuestionRequest):
    question = request_data.question
    
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    try:
        # Use the LangChain chain to generate an answer
        answer = gpu_chain.invoke({"question": question})
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
