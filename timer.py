import requests
import time

url = "http://127.0.0.1:5000/generate-text"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [
        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
        {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}
    ]
}

start_time = time.time()  # Start timer
response = requests.post(url, json=data, headers=headers)
end_time = time.time()  # End timer

response_time = end_time - start_time

if response.status_code == 200:
    print("Response from the server:", response.json())
else:
    print("Error:", response.status_code, response.text)

print(f"Response time: {response_time} seconds")
