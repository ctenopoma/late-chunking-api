import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "cl-nagoya/ruri-v3-310m"
PREFIX = "検索文書: "

class LateChunkingModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    def process(self, document_text: str, chunk_spans: list[tuple[int, int]]):
        # ruri-v3の仕様に合わせてプレフィックスを付与し、スパン位置を補正
        if not document_text.startswith(PREFIX):
            full_text = PREFIX + document_text
            adjusted_spans = [(s + len(PREFIX), e + len(PREFIX)) for s, e in chunk_spans]
        else:
            full_text = document_text
            adjusted_spans = chunk_spans

        # トークナイズ (文字のオフセットマッピングを取得)
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=8192
        )
        
        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 推論 (各トークンのベクトルを取得)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state[0] # (seq_len, hidden_size)

        results = []
        
        # チャンクごとにトークンのベクトルを抽出し、平均化(Mean Pooling)する
        for span_start, span_end in adjusted_spans:
            span_tokens = []
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end: # 特殊トークンはスキップ
                    continue
                # トークンがチャンクの範囲内に含まれているか判定
                if tok_start < span_end and tok_end > span_start:
                    span_tokens.append(token_embeddings[idx])
            
            if not span_tokens:
                # 範囲内にトークンが存在しない場合のフォールバック
                chunk_emb = torch.zeros(token_embeddings.size(-1)).to(self.device)
            else:
                # Mean Pooling
                chunk_emb = torch.stack(span_tokens).mean(dim=0)
            
            # コサイン類似度検索のためにL2正規化
            chunk_emb = torch.nn.functional.normalize(chunk_emb, p=2, dim=0)
            results.append(chunk_emb.cpu().numpy().tolist())

        tokens_used = len(inputs["input_ids"][0])
        return results, tokens_used