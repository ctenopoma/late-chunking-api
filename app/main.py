import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.schemas import LateChunkingRequest, LateChunkingResponse, ChunkEmbedding, Usage
from app.model import LateChunkingModel

# 環境変数からAPIキーを取得
API_KEY = os.getenv("API_KEY", "my-super-secret-api-key")
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

ml_models = {}

# アプリ起動時に1回だけモデルをメモリ/VRAMにロードする
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["late_chunking"] = LateChunkingModel()
    yield
    ml_models.clear()

app = FastAPI(title="Late Chunking API", lifespan=lifespan)

@app.post("/v1/late-chunking", response_model=LateChunkingResponse)
async def late_chunking(request: LateChunkingRequest, api_key: str = Depends(verify_api_key)):
    model = ml_models.get("late_chunking")
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    try:
        embeddings, tokens_used = model.process(request.document_text, request.chunk_spans)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
        
    data = [ChunkEmbedding(index=i, embedding=emb) for i, emb in enumerate(embeddings)]
    return LateChunkingResponse(data=data, usage=Usage(prompt_tokens=tokens_used))