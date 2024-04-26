Run FastAPI server with Python:
```bash
uvicorn hosting_local_llm:app --host 0.0.0.0 --port 5000 --reload

or 

python hosting_local_llm.py
```

Access the server with this command if you would like to use voice function:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate-answer-voice'
```

Access the server with this command if you would like to use the history memory:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate-answer-history' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_question": "I will travel to Singapore, how is the weather there?",
  "session_id": "user123"
}'

curl -X 'POST' \
  'http://127.0.0.1:8000/generate-answer-history' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_question": "Where will I travel to?",
  "session_id": "user123"
}'
```

if you would like to use without the history memory:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/generate-answer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_question": "What is the capital of France?",
  "system_prompt": "You are a friendly chatbot."
}'

```