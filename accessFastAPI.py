import requests

url = 'http://localhost:8000/generate-text'

data = {
        "messages": [
            {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": 'What is the radius of the earth?'},
        ]
}

response = requests.post(url, json=data)
print(response.json())
