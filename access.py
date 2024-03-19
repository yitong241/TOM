import requests
import time

url = "http://127.0.0.1:5000/generate-text"
headers = {"Content-Type": "application/json"}

data = {
        "messages": [
            {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": input()}
        ]
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print(response.json().get("generated_text").split("\n")[5])
    # print("Response from the server:", response.json())
else:
    print("Error:", response.status_code, response.text)

