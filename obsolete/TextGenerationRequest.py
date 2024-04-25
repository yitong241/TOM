from typing import List, Optional
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class TextGenerationRequest(BaseModel):
    messages: Optional[List[Message]] = None
