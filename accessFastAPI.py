import requests

url = 'http://localhost:8000/generate-text'
data = {
    "messages": ["Hello, how are you?"]
}

response = requests.post(url, json=data)
print(response.json())
