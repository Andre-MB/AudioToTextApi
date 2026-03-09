from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
import whisper
import os
import ffmpeg

app = FastAPI()

# rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

model = whisper.load_model("base")

MAX_FILE_SIZE = 3 * 1024 * 1024
MAX_DURATION = 180


def get_duration(file):
    probe = ffmpeg.probe(file)
    return float(probe["format"]["duration"])


@app.post("/transcribe")
@limiter.limit("5/minute")
async def transcribe(request: Request, file: UploadFile = File(...)):

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(400, "Arquivo maior que 3MB")

    path = f"/tmp/{file.filename}"

    with open(path, "wb") as f:
        f.write(contents)

    duration = get_duration(path)

    if duration > MAX_DURATION:
        os.remove(path)
        raise HTTPException(400, "Áudio maior que 3 minutos")

    result = model.transcribe(path)

    os.remove(path)

    return {"text": result["text"]}