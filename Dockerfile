FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
WORKDIR /app

# 必要なパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY ./app ./app

# FastAPIサーバーの起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]