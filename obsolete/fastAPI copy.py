from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
max_length = 256
do_sample = True
temperature = 0.7
top_k = 50
top_p = 0.95
truncation = True

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = 0 if torch.cuda.is_available() else -1
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

def generate_text(question: str, system_prompt_description: str = "You are a friendly chatbot."):
    if not question:
        return "No question provided."
    
    prompt = f"{system_prompt_description}\nQuestion: {question}\n\nAnswer: "
    
    try:
        response = text_generator(prompt, 
                                  max_length=max_length, 
                                  do_sample=do_sample, 
                                  temperature=temperature, 
                                  top_k=top_k, 
                                  top_p=top_p, 
                                  truncation=truncation, 
                                  eos_token_id=tokenizer.eos_token_id)
        
        generated_text = response[0]['generated_text'][len(prompt):]
        answer = generated_text.split("\n")[0]
        return answer.strip()
    except Exception as e:
        return str(e)
