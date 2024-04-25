from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

template = """Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)

gpu_llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    device_map="auto",  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 256, "do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # max_new_tokens: the maximum number of tokens to generate. 
    # In other words, the size of the output sequence, not including the tokens in the prompt. 
    # As an alternative to using the output’s length as a stopping criteria, 
    # you can choose to stop generation whenever the full generation exceeds some amount of time.
)

gpu_chain = prompt | gpu_llm


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
