from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()
VLLM_URL = "http://localhost:5000/generate"

class Query(BaseModel):
    text: str
    max_tokens: int = 256

@app.post("/ask")
def ask(query: Query):
    response = requests.post(VLLM_URL, json={
        "prompt": query.text,
        "max_tokens": query.max_tokens,
        "temperature": 0.3
    })
    return {"answer": response.json()["text"][0]}