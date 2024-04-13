import requests

url = 'http://127.0.0.1:8000/generate-answer'
data = {
    "question": "What is the capital of France?",
    "system_prompt_description": "You are a friendly chatbot."
}

response = requests.post(url, json=data)
assert response.status_code == 200
print(response.json())
