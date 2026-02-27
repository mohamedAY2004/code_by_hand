from pydantic import BaseModel
class ChatRequest(BaseModel):
    message: str
    top_k: int = 3