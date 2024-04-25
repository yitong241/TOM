from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, List
from local_llm import generate_text, generate_text_with_history
import uvicorn
import speech_recognition as sr

class QuestionRequest(BaseModel):
    user_question: Optional[str] = None
    system_prompt: Optional[str] = None

class QuestionRequestHistory(BaseModel):
    user_question: Optional[str] = None
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None

app = FastAPI()

@app.post("/generate-answer")
async def generate_answer(request_data: QuestionRequest):
    user_question = request_data.user_question
    system_prompt = request_data.system_prompt or "You are a friendly chatbot."
    
    if not user_question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    answer = generate_text(user_question, system_prompt)
    
    if answer == "No question provided.":
        raise HTTPException(status_code=400, detail=answer)
    
    return answer

@app.post("/generate-answer-voice")
async def generate_answer_voice():
    system_prompt =  "You are a friendly chatbot."
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        user_question = r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    
    answer = generate_text(user_question, system_prompt)
    
    if answer == "No question provided.":
        raise HTTPException(status_code=400, detail=answer)
    
    return answer




history_store: Dict[str, List[dict]] = {}

@app.post("/generate-answer-history")
async def generate_answer_history(request_data: QuestionRequestHistory):
    user_question = request_data.user_question
    system_prompt = request_data.system_prompt or "You are a friendly chatbot."
    session_id = request_data.session_id or "default_session"
    
    if not user_question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    history = history_store.get(session_id, [])
    
    answer = generate_text_with_history(user_question, history, system_prompt)
    history_store[session_id] = history
    
    if answer == "No question provided.":
        raise HTTPException(status_code=400, detail=answer)
    
    return answer

if __name__ == "__main__":
    uvicorn.run("hosting_local_llm:app", host="0.0.0.0", port=8000, reload=True)
