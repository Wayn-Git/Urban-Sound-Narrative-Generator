# Backend v2.0.5 - Enhanced CORS for Audio Streaming

import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import torchaudio
import whisper
from panns_inference import AudioTagging, labels
import requests
from groq import Groq
from pydub import AudioSegment
from pathlib import Path
import warnings
import uuid
import json
import asyncio
import logging
from typing import List, Tuple
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

models = {}

GROQ_MODEL_CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.7,
    "max_tokens": 150,
    "top_p": 0.9,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.2
}

ELEVENLABS_V3_CONFIG = {
    "voice_id": "cgSgspJ2msm6clMCkdW9",
    "model_id": "eleven_turbo_v2_5",
    "voice_settings": {
        "stability": 0.4,
        "similarity_boost": 0.75,
        "style": 0.8,
        "use_speaker_boost": True
    },
    "output_format": "mp3_44100_128"
}

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

async def load_whisper_lazy():
    if 'whisper' in models and models['whisper'] is not None:
        return models['whisper']
    max_retries = 3
    retry_delay = 10
    for attempt in range(max_retries):
        try:
            logger.info(f"Lazy-loading Whisper model (attempt {attempt + 1}/{max_retries})...")
            models['whisper'] = whisper.load_model("base", device='cpu', download_root='/root/.cache/whisper')
            logger.info("Whisper model loaded successfully")
            return models['whisper']
        except Exception as e:
            logger.warning(f"Failed to load Whisper (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Whisper load failed after {max_retries} attempts: {str(e)}")
                models['whisper'] = None
                return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI models...")
    try:
        models['panns'] = AudioTagging(checkpoint_path=None, device='cpu')
        logger.info("PANNs model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load PANNs model: {str(e)}")
        raise
    try:
        await load_whisper_lazy()
    except Exception as e:
        logger.warning(f"Initial Whisper load failed (will retry lazily): {str(e)}")
        models['whisper'] = None
    os.makedirs("audio", exist_ok=True)
    logger.info("Models setup complete. Server ready!")
    yield
    models.clear()
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Urban Sound Narrative API",
    description="AI-Powered Audio Processing for Emotionally Rich Narration",
    version="2.0.5",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # CRITICAL: Expose all headers for audio streaming
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_optimized_groq_prompt(sound_context: str, transcript: str) -> str:
    prompt_text = (
        "You are a master storyteller crafting a cinematic urban narrative. "
        f"Based on these detected sounds: {sound_context}, write a vivid, "
        "emotionally evocative 2-3 sentence scene that immerses the reader in a bustling city moment.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        f"1. Seamlessly integrate this spoken phrase into the scene: '{transcript}'\n"
        "2. Use square brackets to mark emotional delivery ONLY for dialogue, like: "
        "[enthusiastically], [wearily], [sarcastically], [thoughtfully]\n"
        "3. Make the scene feel alive - use sensory details, movement, and atmosphere\n"
        "4. DO NOT include sound effect descriptions like (car honking) or *dog barks*\n"
        "5. Focus on human moments and urban poetry\n\n"
        "Example style: 'The afternoon pulse of the city beats steadily as vendors call out their wares. "
        'A passerby mutters [sarcastically], "Another beautiful day in paradise," while dodging between rushing commuters.\''
    )
    return prompt_text

def build_optimized_elevenlabs_prompt(narration_text: str, sound_context: str) -> str:
    return f"""<voice>{narration_text}</voice>
<style>
Deliver this urban narrative with genuine emotion and theatrical flair. Use dynamic pacing: speak deliberately during descriptive passages to build atmosphere, then shift to natural conversational energy for any dialogue. Infuse warmth and immersion into environmental descriptions. For bracketed emotional cues like [enthusiastically] or [thoughtfully], embody that emotion fully - don't just read the word, become it. Vary your pitch and intensity to match the scene's energy. Make the listener feel present in this urban moment.
</style>
<context>
This is a vivid narration of an urban soundscape featuring: {sound_context}. The scene should feel cinematic and emotionally resonant, transporting the listener into a real city moment with all its texture and humanity.
</context>"""

async def generate_narrative_text(sound_types: List[str], transcript: str) -> Tuple[str, str]:
    try:
        sound_context = ", ".join(sound_types)
        logger.info(f"Generating narrative for sounds: {sound_context}")
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_optimized_groq_prompt(sound_context, transcript)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **GROQ_MODEL_CONFIG
        )
        narration = response.choices[0].message.content.strip()
        if not narration:
            raise ValueError("GROQ returned empty narration")
        logger.info(f"Generated narration ({len(narration)} chars)")
        return narration, sound_context
    except Exception as e:
        logger.error(f"GROQ API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate narration: {str(e)}")

async def synthesize_audio(narration_text: str, sound_context: str, output_filename: str) -> str:
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
        elevenlabs_prompt = build_optimized_elevenlabs_prompt(narration_text, sound_context)
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_V3_CONFIG['voice_id']}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": elevenlabs_prompt,
            "model_id": ELEVENLABS_V3_CONFIG["model_id"],
            "voice_settings": ELEVENLABS_V3_CONFIG["voice_settings"],
            "output_format": ELEVENLABS_V3_CONFIG["output_format"]
        }
        logger.info(f"Synthesizing speech with V3 (voice: {ELEVENLABS_V3_CONFIG['voice_id']})")
        tts_response = requests.post(url, headers=headers, json=payload, timeout=45)
        if tts_response.status_code != 200:
            error_detail = tts_response.text
            logger.error(f"ElevenLabs API error [{tts_response.status_code}]: {error_detail}")
            raise HTTPException(
                status_code=tts_response.status_code,
                detail=f"ElevenLabs TTS failed: {error_detail}"
            )
        logger.info(f"TTS synthesis successful ({len(tts_response.content)} bytes)")
        output_path = f"audio/{output_filename}"
        temp_path = f"audio/temp_{output_filename}"
        with open(temp_path, "wb") as f:
            f.write(tts_response.content)
        logger.info("Applying audio normalization and enhancement")
        audio = AudioSegment.from_mp3(temp_path)
        audio = audio + 6
        audio = audio.normalize()
        audio = audio.fade_in(100).fade_out(200)
        audio.export(output_path, format="mp3", bitrate="192k", parameters=["-q:a", "0"])
        Path(temp_path).unlink(missing_ok=True)
        logger.info(f"Audio exported to {output_path}")
        return output_path
    except requests.exceptions.Timeout:
        logger.error("ElevenLabs API timeout")
        raise HTTPException(status_code=504, detail="Text-to-speech service timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"ElevenLabs request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        Path(f"audio/temp_{output_filename}").unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

async def process_audio_file_streaming(audio_path: str, output_filename: str):
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Upload file not found: {audio_path}")
            yield json.dumps({"stage": "error", "message": "Upload file not found", "sounds": [], "progress": 0}) + "\n"
            return
        file_size = os.path.getsize(audio_path)
        logger.info(f"Processing audio: {audio_path} ({file_size} bytes)")
        yield json.dumps({"stage": "loading", "message": "🎧 Analyzing your audio...", "sounds": [], "progress": 10}) + "\n"
        await asyncio.sleep(0.5)
        yield json.dumps({"stage": "transcribing", "message": "👂 Listening to speech...", "sounds": [], "progress": 20}) + "\n"
        whisper_model = await load_whisper_lazy()
        if whisper_model is None:
            logger.warning("Whisper unavailable - using fallback transcript")
            transcript = "(speech detected but transcription unavailable)"
            yield json.dumps({"stage": "transcribe_warn", "message": "⚠️ Transcription fallback", "sounds": [], "progress": 30}) + "\n"
        else:
            try:
                result = whisper_model.transcribe(audio_path)
                transcript = result["text"].strip() or "(no speech detected)"
                logger.info(f"Transcribed: {transcript}")
                yield json.dumps({"stage": "transcribe_complete", "message": "✅ Speech transcribed", "sounds": [], "progress": 30}) + "\n"
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                transcript = "(transcription failed)"
                yield json.dumps({"stage": "transcribe_warn", "message": "⚠️ Transcription failed", "sounds": [], "progress": 30}) + "\n"
        await asyncio.sleep(0.3)
        yield json.dumps({"stage": "extracting", "message": "🔊 Extracting sound patterns...", "sounds": [], "progress": 40}) + "\n"
        waveform, sr = torchaudio.load(audio_path)
        if sr != 32000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.to(device)
        await asyncio.sleep(0.5)
        yield json.dumps({"stage": "identifying", "message": "🎵 Identifying urban soundscapes...", "sounds": [], "progress": 50}) + "\n"
        clipwise_output, _ = models['panns'].inference(waveform)
        clipwise_output = clipwise_output.squeeze()
        top_indices = clipwise_output.argsort()[-5:][::-1]
        sound_types = [labels[int(i)] for i in top_indices]
        logger.info(f"Detected sounds: {sound_types}")
        yield json.dumps({"stage": "sounds_detected", "message": "✨ Sounds detected!", "sounds": sound_types, "progress": 60}) + "\n"
        try:
            Path(audio_path).unlink()
            logger.info(f"Cleaned up input file: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not delete input file: {e}")
        await asyncio.sleep(0.8)
        yield json.dumps({"stage": "ai_processing", "message": "🤖 Crafting cinematic narrative...", "sounds": sound_types, "progress": 70}) + "\n"
        await asyncio.sleep(0.5)
        narration, sound_context = await generate_narrative_text(sound_types, transcript)
        logger.info(f"Narration generated: {narration[:100]}...")
        yield json.dumps({"stage": "narrating", "message": "✍️ Preparing emotional delivery...", "sounds": sound_types, "progress": 80}) + "\n"
        await asyncio.sleep(0.5)
        yield json.dumps({"stage": "voice_generation", "message": "🎙️ Generating expressive narration...", "sounds": sound_types, "progress": 90}) + "\n"
        output_path = await synthesize_audio(narration, sound_context, output_filename)
        yield json.dumps({"stage": "finalizing", "message": "✨ Adding final touches...", "sounds": sound_types, "progress": 95}) + "\n"
        await asyncio.sleep(0.5)
        if not os.path.exists(output_path):
            logger.error(f"Output audio file not created: {output_path}")
            yield json.dumps({"stage": "error", "message": "Failed to create audio file", "sounds": [], "progress": 0}) + "\n"
            return
        audio_url = f"/audio/{output_filename}"
        logger.info(f"Audio available at: {audio_url}")
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
        logger.error(f"Streaming processing error: {str(e)}", exc_info=True)
        yield json.dumps({"stage": "error", "message": f"❌ Error: {str(e)}", "sounds": [], "progress": 0}) + "\n"

@app.post("/process-audio-stream")
async def process_audio_stream(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    unique_id = str(uuid.uuid4())
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    output_filename = f"output_{unique_id}.mp3"
    audio_path = f"audio/upload_{unique_id}_{safe_filename}"
    try:
        content = await file.read()
        with open(audio_path, "wb") as f:
            f.write(content)
        logger.info(f"Uploaded {len(content)} bytes to {audio_path}")
        async def cleanup_output():
            await asyncio.sleep(300)
            output_path = f"audio/{output_filename}"
            if os.path.exists(output_path):
                Path(output_path).unlink()
                logger.info(f"Output cleaned: {output_path}")
        background_tasks.add_task(cleanup_output)
        return StreamingResponse(
            process_audio_file_streaming(audio_path, output_filename),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Streaming endpoint error: {str(e)}", exc_info=True)
        Path(audio_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio file with proper headers for streaming playback"""
    file_path = f"audio/{filename}"
    if not os.path.exists(file_path):
        logger.error(f"Audio file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")
    logger.info(f"Serving audio file: {file_path}")
    
    # Return with explicit headers for audio streaming
    return FileResponse(
        file_path, 
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "*"
        }
    )

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Urban Sound Narrative API v2.0.5 (Enhanced Audio Streaming)",
        "version": "2.0.5",
        "optimizations": [
            "GROQ Llama 3.3 70B with creative parameters",
            "ElevenLabs V3 Alpha emotional prompting",
            "Enhanced audio post-processing",
            "Lazy Whisper loading for network resilience",
            "Filename sanitization for FFmpeg compatibility",
            "Fixed processing order: transcribe → sounds → delete input",
            "Enhanced CORS headers for audio streaming"
        ]
    }

@app.get("/health")
async def health_check():
    whisper_status = "loaded" if models.get('whisper') is not None else "lazy-pending"
    return {
        "status": "healthy",
        "models_loaded": len([k for k, v in models.items() if v is not None]),
        "device": device,
        "groq_configured": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL_CONFIG["model"],
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "elevenlabs_model": ELEVENLABS_V3_CONFIG["model_id"],
        "voice_id": ELEVENLABS_V3_CONFIG["voice_id"],
        "whisper_status": whisper_status
    }

os.makedirs("audio", exist_ok=True)