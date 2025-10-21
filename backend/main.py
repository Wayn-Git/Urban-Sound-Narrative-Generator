import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
import whisper
from panns_inference import AudioTagging, labels
import numpy as np
import requests
from groq import Groq
from pydub import AudioSegment
from pathlib import Path

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.vercel.app"],  # Update with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Response model
class AudioResponse(BaseModel):
    narration: str
    audio_path: str

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
panns_model = AudioTagging(checkpoint_path=None, device=device)
whisper_model = whisper.load_model("base", device=device)

def process_audio(audio_path: str):
    # Load and prepare audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 32000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)

    # Get sound tags
    clipwise_output, _ = panns_model.inference(waveform)
    clipwise_output = clipwise_output.squeeze()
    top_indices = clipwise_output.argsort()[-5:][::-1]
    sound_types = [labels[int(i)] for i in top_indices]
    sound_str = ", ".join(sound_types)

    # Transcribe speech
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"].strip() or "(no speech)"

    # Generate narration with Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        f"Write a lively, conversational 2-3 sentence narration of a bustling city scene, capturing the urban vibe based on sounds: {sound_str}. "
        f"Seamlessly integrate a passerby or vendor casually saying '{transcript}' as part of the scene, using square brackets (e.g., [thoughtfully], [sarcastically], [enthusiastically]) to denote emotional voice tones for expressive dialogue delivery. "
        f"Ensure the narration is natural, immersive, and reflects the full essence of the audio through vivid emotional expression, without including sound effect descriptions in the text."
    )
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=100
    )
    narration = response.choices[0].message.content

    # Convert to speech with ElevenLabs
    url = "https://api.elevenlabs.io/v1/text-to-speech/cgSgspJ2msm6clMCkdW9"  # Jessica voice
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": narration,
        "model_id": "eleven_multilingual_v2",  # Fallback due to eleven_v3 issues
        "voice_settings": {
            "stability": 0.0,  # Creative mode for emotions
            "similarity_boost": 0.85,
            "style": 0.6,
            "use_speaker_boost": True
        }
    }

    output_path = "audio/output.mp3"
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            temp_file = "audio/temp_output.mp3"
            os.makedirs("audio", exist_ok=True)
            with open(temp_file, "wb") as f:
                f.write(response.content)
            audio = AudioSegment.from_mp3(temp_file)
            audio = audio + 8
            audio = audio.normalize()
            audio.export(output_path, format="mp3", bitrate="192k")
            Path(temp_file).unlink(missing_ok=True)
            return narration, output_path
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.on_event("startup")
async def load_models():
    global panns_model, whisper_model
    panns_model = AudioTagging(checkpoint_path=None, device='cpu')
    whisper_model = whisper.load_model("base", device='cpu')

@app.post("/process-audio", response_model=AudioResponse)
async def process_audio(file: UploadFile = File(...)):
    # Validate file
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    # Save uploaded file
    audio_path = f"audio/{file.filename}"
    os.makedirs("audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(await file.read())
    
    # Process audio
    try:
        narration, output_path = process_audio(audio_path)
        return AudioResponse(narration=narration, audio_path=output_path)
    finally:
        Path(audio_path).unlink(missing_ok=True)  # Clean up uploaded file

# Serve audio files
app.mount("/audio", StaticFiles(directory="audio"), name="audio")