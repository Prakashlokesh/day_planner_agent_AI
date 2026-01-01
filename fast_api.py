from fastapi import FastAPI
from pydantic import BaseModel
from graph import build_graph
from memory import init_db

app = FastAPI()
graph = build_graph()
init_db()

class ChatIn(BaseModel):
    user_id: str = "default"
    message: str

@app.post("/chat")
def chat(payload: ChatIn):
    result = graph.invoke({"user_id": payload.user_id, "user_request": payload.message})
    return {
        "notes": result.get("notes", []),
        "weekly_schedule": result.get("weekly_schedule", {})
    }
