from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
import whisper
import os
import ffmpeg

os.environ["PATH"] += r";C:\Users\André\ffmpeg\bin"

app = FastAPI()

# Rate limit
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# modelo whisper
model = whisper.load_model("base")

MAX_FILE_SIZE = 3 * 1024 * 1024
MAX_DURATION = 180


def get_audio_duration(file):
    probe = ffmpeg.probe(file)
    return float(probe["format"]["duration"])


@app.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe(request: Request, file: UploadFile = File(...)):

    contents = await file.read()

    # limitar tamanho
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Arquivo maior que 3MB")

    path = f"temp_{file.filename}"
    
    with open(path, "wb") as f:
        f.write(contents)

    # verificar duração
    duration = get_audio_duration(path)

    if duration > MAX_DURATION:
        os.remove(path)
        raise HTTPException(status_code=400, detail="Áudio maior que 3 minutos")

    # transcrição
    result = model.transcribe(path)

    # apagar arquivo
    os.remove(path)

    return {
        "filename": file.filename,
        "duration": duration,
        "text": result["text"]
    }