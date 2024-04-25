import requests

def generate_answer(user_question, system_prompt="You are a friendly chatbot."):
    API_URL = "http://localhost:5000/generate-answer"
    data = {
        "user_question": user_question,
        "system_prompt": system_prompt,
    }
    response = requests.post(API_URL, json=data)
    return response.json()

def generate_answer_with_history(user_question, system_prompt="You are a friendly chatbot.", session_id=None):
    API_URL = "http://localhost:5000/generate-answer-history"
    data = {
        "user_question": user_question,
        "system_prompt": system_prompt,
        "session_id": session_id
    }
    response = requests.post(API_URL, json=data)
    return response.json()

if __name__ == "__main__":
    user_question = "What is the capital of France?"
    system_prompt = "You are a friendly chatbot."
    response = generate_answer(user_question, system_prompt)
    print(response)