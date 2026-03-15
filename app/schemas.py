from pydantic import BaseModel, Field
from typing import List, Tuple

class LateChunkingRequest(BaseModel):
    document_text: str = Field(..., description="ベクトル化する長文ドキュメント全体")
    chunk_spans: List[Tuple[int, int]] = Field(..., description="各チャンクの開始・終了文字インデックスのリスト")

class Usage(BaseModel):
    prompt_tokens: int

class ChunkEmbedding(BaseModel):
    index: int
    embedding: List[float]

class LateChunkingResponse(BaseModel):
    data: List[ChunkEmbedding]
    usage: Usage