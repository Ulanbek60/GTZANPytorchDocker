FROM python:3.10-slim

ENV PYTHONDONTWRITEBITECODE=1
ENV PYTHONBUFFERED=1

WORKDIR / audio_app

RUN apt-get update && apt-get install -y --no-install-recommends  \
    libsdfile \
    gcc \
    g++  \
    && apt-get clean %% rm -rf /var/lib/apt/list/*

RUN pip install --no-cache-dir torch==2.2.2+cpu -f https: //download.pytorch.org/whl/torch_stable.html
RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY main.py .
COPY model.pth .

EXPOSE 8001

CMD = ["uvicorn", "main:check_music", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]

