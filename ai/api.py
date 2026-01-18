from fastapi import FastAPI
from sync_drive import sync_drive
from ingest_db import ingest
from fastapi import HTTPException
from dotenv import load_dotenv
from pathlib import Path
from chat import chat_stream
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from databaselib import get_messages
import pickle
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer


with open("graph.pkl", "rb") as f:
        G = pickle.load(f)
node_embeddings = np.load("node_embeddings.npy")
node_ids = None
with open("node_ids.json") as f:
    node_ids = json.load(f)


# ---- load .env ----
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    guardrail_enabled:bool
    rag_mode: str


# ---- Google Drive ----
# ---- API ----
@app.post("/sync-drive")
def sync_drive_api():
    try:
        return sync_drive()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_api():
    try:
        return ingest()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_api(req: ChatRequest):
    user_input = req.message
    guardrail_enabled = req.guardrail_enabled
    rag_mode = req.rag_mode
    
    return await chat_stream(
        user_input, 
        guardrail_enabled, 
        G, 
        node_ids, 
        node_embeddings, 
        rag_mode
    )
    
@app.get("/messages")
def get_messages_api(limit: int = 50):
    try:
        return get_messages(limit)
    except Exception as e:
        print(e)
        return e

