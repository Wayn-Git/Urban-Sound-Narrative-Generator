import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import torchaudio
import whisper
from panns_inference import AudioTagging, labels
import numpy as np
import requests
from groq import Groq
from pydub import AudioSegment
from pathlib import Path
import warnings
import uuid
import json
import asyncio
from typing import List

# Suppress pydub regex warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Global model storage
models = {}

# Response models
class ProcessingUpdate(BaseModel):
    stage: str
    message: str
    sounds: List[str] = []
    progress: int

class AudioResponse(BaseModel):
    narration: str
    audio_url: str
    detected_sounds: List[str]
    transcript: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup: Load models
    print("Loading models...")
    models['panns'] = AudioTagging(checkpoint_path=None, device='cpu')
    models['whisper'] = whisper.load_model("base", device='cpu')
    print("Models loaded successfully")
    
    # Create audio directory
    os.makedirs("audio", exist_ok=True)
    
    yield
    
    # Shutdown: Cleanup if needed
    models.clear()
    print("Application shutdown")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Urban Sound Narrative API",
    description="Audio processing API for urban sound narration",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'


async def process_audio_file_streaming(audio_path: str, output_filename: str):
    """Process audio file and yield progress updates"""
    
    try:
        # Stage 1: Load and prepare audio
        yield json.dumps({
            "stage": "loading",
            "message": "🎧 Analyzing your audio...",
            "sounds": [],
            "progress": 10
        }) + "\n"
        await asyncio.sleep(0.5)
        
        waveform, sr = torchaudio.load(audio_path)
        if sr != 32000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(device)

        # Stage 2: Extract sound patterns
        yield json.dumps({
            "stage": "extracting",
            "message": "🔊 Extracting sound patterns...",
            "sounds": [],
            "progress": 25
        }) + "\n"
        await asyncio.sleep(0.5)

        # Stage 3: Get sound tags
        yield json.dumps({
            "stage": "identifying",
            "message": "🎵 Identifying urban soundscapes...",
            "sounds": [],
            "progress": 40
        }) + "\n"
        
        clipwise_output, _ = models['panns'].inference(waveform)
        clipwise_output = clipwise_output.squeeze()
        top_indices = clipwise_output.argsort()[-5:][::-1]
        sound_types = [labels[int(i)] for i in top_indices]
        sound_str = ", ".join(sound_types)

        # Stage 4: Show detected sounds
        yield json.dumps({
            "stage": "sounds_detected",
            "message": "✨ Sounds detected!",
            "sounds": sound_types,
            "progress": 50
        }) + "\n"
        await asyncio.sleep(0.8)

        # Stage 5: Transcribe speech
        yield json.dumps({
            "stage": "transcribing",
            "message": "👂 Listening to speech...",
            "sounds": sound_types,
            "progress": 60
        }) + "\n"
        
        result = models['whisper'].transcribe(audio_path)
        transcript = result["text"].strip() or "(no speech)"

        # Stage 6: Generate narration with Groq
        yield json.dumps({
            "stage": "ai_processing",
            "message": "🤖 Feeding sounds to AI brain...",
            "sounds": sound_types,
            "progress": 70
        }) + "\n"
        await asyncio.sleep(0.5)

        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        
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

        # Stage 7: Generate voice
        yield json.dumps({
            "stage": "narrating",
            "message": "✍️ Crafting your narrative...",
            "sounds": sound_types,
            "progress": 80
        }) + "\n"
        await asyncio.sleep(0.5)

        # Stage 8: Text to speech
        yield json.dumps({
            "stage": "voice_generation",
            "message": "🎙️ Generating voice narration...",
            "sounds": sound_types,
            "progress": 90
        }) + "\n"

        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
        
        url = "https://api.elevenlabs.io/v1/text-to-speech/cgSgspJ2msm6clMCkdW9"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": narration,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.0,
                "similarity_boost": 0.85,
                "style": 0.6,
                "use_speaker_boost": True
            }
        }

        output_path = f"audio/{output_filename}"
        tts_response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if tts_response.status_code == 200:
            temp_file = f"audio/temp_{output_filename}"
            with open(temp_file, "wb") as f:
                f.write(tts_response.content)
            audio = AudioSegment.from_mp3(temp_file)
            audio = audio + 8
            audio = audio.normalize()
            audio.export(output_path, format="mp3", bitrate="192k")
            Path(temp_file).unlink(missing_ok=True)
        else:
            raise HTTPException(
                status_code=tts_response.status_code, 
                detail=f"ElevenLabs API error: {tts_response.text}"
            )

        # Stage 9: Finalizing
        yield json.dumps({
            "stage": "finalizing",
            "message": "✨ Adding final touches...",
            "sounds": sound_types,
            "progress": 95
        }) + "\n"
        await asyncio.sleep(0.5)

        # Stage 10: Complete
        audio_url = f"/audio/{output_filename}"
        yield json.dumps({
            "stage": "complete",
            "message": "✅ Complete!",
            "sounds": sound_types,
            "progress": 100,
            "narration": narration,
            "audio_url": audio_url,
            "transcript": transcript
        }) + "\n"

    except Exception as e:
        yield json.dumps({
            "stage": "error",
            "message": f"❌ Error: {str(e)}",
            "sounds": [],
            "progress": 0
        }) + "\n"


def process_audio_file_sync(audio_path: str, output_filename: str) -> tuple[str, str, List[str], str]:
    """Synchronous version for non-streaming endpoint"""
    
    # Load and prepare audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 32000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.to(device)

    # Get sound tags
    clipwise_output, _ = models['panns'].inference(waveform)
    clipwise_output = clipwise_output.squeeze()
    top_indices = clipwise_output.argsort()[-5:][::-1]
    sound_types = [labels[int(i)] for i in top_indices]
    sound_str = ", ".join(sound_types)

    # Transcribe speech
    result = models['whisper'].transcribe(audio_path)
    transcript = result["text"].strip() or "(no speech)"

    # Generate narration with Groq
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        f"Write a lively, conversational 2-3 sentence narration of a bustling city scene, capturing the urban vibe based on sounds: {sound_str}. "
        f"Seamlessly integrate a passerby or vendor casually saying '{transcript}' as part of the scene, using square brackets (e.g., [thoughtfully], [sarcastically], [enthusiastically]) to denote emotional voice tones for expressive dialogue delivery. "
        f"Ensure the narration is natural, immersive, and reflects the full essence of the audio through vivid emotional expression, without including sound effect descriptions in the text."
    )
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=100
        )
        narration = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API failed: {str(e)}")

    # Convert to speech with ElevenLabs
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
    
    url = "https://api.elevenlabs.io/v1/text-to-speech/cgSgspJ2msm6clMCkdW9"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg"
    }
    payload = {
        "text": narration,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.0,
            "similarity_boost": 0.85,
            "style": 0.6,
            "use_speaker_boost": True
        }
    }

    output_path = f"audio/{output_filename}"
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            temp_file = f"audio/temp_{output_filename}"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            audio = AudioSegment.from_mp3(temp_file)
            audio = audio + 8
            audio = audio.normalize()
            audio.export(output_path, format="mp3", bitrate="192k")
            Path(temp_file).unlink(missing_ok=True)
            return narration, output_path, sound_types, transcript
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"ElevenLabs API error: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"TTS request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")


@app.post("/process-audio", response_model=AudioResponse)
async def process_audio_endpoint(file: UploadFile = File(...)):
    """Process uploaded audio file - synchronous version"""
    # Validate file
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    output_filename = f"output_{unique_id}.mp3"
    audio_path = f"audio/upload_{unique_id}_{file.filename}"
    os.makedirs("audio", exist_ok=True)
    
    try:
        # Save uploaded file
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process audio
        narration, output_path, sound_types, transcript = process_audio_file_sync(audio_path, output_filename)
        
        # Return URL instead of path
        audio_url = f"/audio/{output_filename}"
        return AudioResponse(
            narration=narration,
            audio_url=audio_url,
            detected_sounds=sound_types,
            transcript=transcript
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        Path(audio_path).unlink(missing_ok=True)


@app.post("/process-audio-stream")
async def process_audio_stream(file: UploadFile = File(...)):
    """Process uploaded audio file with streaming progress updates"""
    # Validate file
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    output_filename = f"output_{unique_id}.mp3"
    audio_path = f"audio/upload_{unique_id}_{file.filename}"
    os.makedirs("audio", exist_ok=True)
    
    try:
        # Save uploaded file
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Return streaming response
        return StreamingResponse(
            process_audio_file_streaming(audio_path, output_filename),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup will happen after streaming completes
        await asyncio.sleep(1)
        Path(audio_path).unlink(missing_ok=True)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files"""
    file_path = f"audio/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Urban Sound Narrative API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) == 2,
        "device": device,
        "groq_configured": bool(GROQ_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY)
    }


# Mount static files directory for audio
os.makedirs("audio", exist_ok=True)