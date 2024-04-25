Run FastAPI server with Python:
```bash
python -m uvicorn fastAPI:app --reload
```

Access the server with command:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate-answer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is the capital of France?",
  "system_prompt_description": "You are a friendly chatbot."
}'

```



Run FastAPI server with Python and LangChain
:
```bash
python -m uvicorn langChain:app --reload
```

Access the server with command and example quetion:
```bash
curl -X 'POST' \
  'http://localhost:8000/generate-answer' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is electroencephalography?"}'
```



Running TinyLlama server with Python:

```bash
python main.py
```

Access the server with:

```bash
curl -X POST 127.0.0.1:5000/generate-text \
-H "Content-Type: application/json" \
-d "{\"messages\": [{\"role\": \"system\", \"content\": \"You are a friendly chatbot.\"}, {\"role\": \"user\", \"content\": \"<Input Question>\"}]}"
```

or with the Python script:

```bash
python access.py
```
and key in the prompt that you have.


