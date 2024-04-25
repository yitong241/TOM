from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from collections import deque
import json

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 1024
'''
max_new_tokens: the maximum number of tokens to generate. 
In other words, the size of the output sequence, not including the tokens in the prompt. 
As an alternative to using the output’s length as a stopping criteria, you can choose to stop generation whenever the full generation exceeds some amount of time.
If max_new_tokens is set too low: The generated output will be shorter, and the model will aim to conclude the text within the specified token limit. 
This means that if the limit is reached mid-sentence, the model might attempt to end the sentence early, potentially resulting in an abrupt or incomplete sentence. 
The model tries to generate text that makes sense up to the limit, so it may use shorter words or phrases to fit within the constraint.
'''
DO_SAMPLE = True
'''
do_sample=False: The model deterministically picks the most probable next token at each step. 
    This generally results in more predictable and repetitive text because the model will always choose the most likely word or punctuation. This method is also referred to as "greedy" decoding.
do_sample=True: The model samples from the probability distribution of the tokens, which can lead to more diverse and interesting outputs. 
    This method allows the model to generate less predictable and more varied text, as it can choose less likely tokens based on their assigned probabilities.
'''
TEMPERATURE = 0.7
TOP_K = 50
'''
text generation process is restricted to randomly selecting the next token from the top most likely next tokens as predicted by the language model. 
The distribution from which the token is sampled is limited to these top k choices, which can include less probable options within that subset.
'''
TOP_P = 0.95
'''
When top_p is set, the model will select the smallest possible set of tokens whose cumulative probability adds up to p or more. 
This set can be thought of as a nucleus of tokens from which the model will sample the next token. 
'''
TRUNCATION = True

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model, device=device, torch_dtype=torch.bfloat16, device_map="auto", tokenizer=tokenizer)

def generate_text(user_question: str, system_prompt: str = "You are a friendly chatbot."):
    if not user_question:
        return "No question provided."
    
    messages = [
    {
        "role": "system",
        "content": system_prompt,
        },
    {
        "role": "user", 
        "content": user_question
        },
    ]
    
    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt,
                       max_length=MAX_LENGTH,
                       do_sample=DO_SAMPLE,
                       temperature=TEMPERATURE,
                       top_k=TOP_K,
                       top_p=TOP_P,
                       truncation=TRUNCATION)
        output_text = outputs[0]["generated_text"]
        return output_text
    except Exception as e:
        return str(e)
    
def generate_text_with_history(user_question: str, history: list, system_prompt: str = "You are a friendly chatbot."):
    if not user_question:
        return "No question provided."
    history.append({"role": "user", "content": user_question})

    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_length=MAX_LENGTH, do_sample=DO_SAMPLE, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, truncation=TRUNCATION)
        
        # Append generated response to history
        history.append({"role": "system", "content": outputs[0]["generated_text"]})
        output_text = outputs[0]["generated_text"]
        return output_text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    user_question = "What is the capital of France?"
    system_prompt = "You are a friendly chatbot."
    response = generate_text(user_question, system_prompt)
    print(response)
    response = generate_text_with_history(user_question, [{"role": "user", "content": "What is the capital of France?"}], system_prompt)
    print(response)
    