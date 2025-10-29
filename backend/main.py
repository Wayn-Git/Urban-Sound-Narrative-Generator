import os
import whisper
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment
from panns_inference import AudioTagging, labels
from pathlib import Path
import logging, asyncio, json, uuid
from fastapi.responses import StreamingResponse, FileResponse

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="Urban Sound Narrative API", version="2.0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("audio", exist_ok=True)
models = {}

# ---------- Lazy Model Load ----------
async def load_whisper_lazy():
    if "whisper" not in models:
        logger.info("Loading Whisper model...")
        models["whisper"] = whisper.load_model("base", device="cpu")
    return models["whisper"]

@app.on_event("startup")
async def load_models():
    logger.info("Loading PANNs...")
    models["panns"] = AudioTagging(checkpoint_path=None, device="cpu")

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4().hex}.wav"
    file_path = f"audio/{filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    whisper_model = await load_whisper_lazy()
    transcript = whisper_model.transcribe(file_path)["text"].strip()
    waveform, sr = torchaudio.load(file_path)
    if sr != 32000:
        waveform = torchaudio.transforms.Resample(sr, 32000)(waveform)
    clipwise_output, _ = models["panns"].inference(waveform)
    top_indices = clipwise_output.squeeze().argsort()[-5:][::-1]
    sound_types = [labels[int(i)] for i in top_indices]

    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"Urban narrative about {', '.join(sound_types)} with speech '{transcript}'"
    groq_resp = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=150,
    )
    narration = groq_resp.choices[0].message.content.strip()

    # ElevenLabs synthesis
    url = f"https://api.elevenlabs.io/v1/text-to-speech/cgSgspJ2msm6clMCkdW9"
    headers = {
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": narration,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75,
            "style": 0.8,
            "use_speaker_boost": True,
        },
    }
    import requests
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=r.text)
    output_path = f"audio/{filename}.mp3"
    with open(output_path, "wb") as out:
        out.write(r.content)
    return {
        "narration": narration,
        "transcript": transcript,
        "sounds": sound_types,
        "audio_url": f"/audio/{filename}.mp3",
    }

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    path = Path("audio") / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
