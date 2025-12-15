FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /audio_app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    gcc \
    g++ \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    torchaudio==2.2.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY main.py .
COPY model.pth .

EXPOSE 8001

CMD ["uvicorn", "main:check_music", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
