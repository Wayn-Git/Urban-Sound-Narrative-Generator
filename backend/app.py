# Optimized Backend v3.0.0 - Memory-Efficient Architecture
# Designed for deployment on Render, Railway, or similar platforms

import os
import gc
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager
import asyncio
import uuid
import re

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import torchaudio
import requests
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ============= MODEL CONFIGURATION =============
# Use smaller, faster models to reduce memory footprint
WHISPER_MODEL_SIZE = "tiny"  # Options: tiny (39MB), base (74MB), small (244MB)
DEVICE = 'cpu'  # Force CPU to avoid CUDA memory issues on free tiers

# Global model cache (loaded on-demand, cleared after use)
model_cache = {
    "whisper": None,
    "panns": None,
    "whisper_last_used": 0,
    "panns_last_used": 0
}

# Model timeout (seconds) - unload if not used
MODEL_TIMEOUT = 300  # 5 minutes

# ============= API CONFIGURATIONS =============
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

# ============= PYDANTIC MODELS =============
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

# ============= MEMORY-EFFICIENT MODEL LOADING =============

async def load_whisper_model() -> Optional[object]:
    """
    Lazy-load Whisper model with aggressive memory management.
    Uses smallest model variant for minimal footprint.
    """
    import time
    
    if model_cache["whisper"] is not None:
        model_cache["whisper_last_used"] = time.time()
        logger.info("Using cached Whisper model")
        return model_cache["whisper"]
    
    try:
        logger.info(f"Loading Whisper {WHISPER_MODEL_SIZE} model (CPU)...")
        import whisper
        
        model = whisper.load_model(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            download_root='/tmp/.cache/whisper'  # Use /tmp for ephemeral storage
        )
        
        model_cache["whisper"] = model
        model_cache["whisper_last_used"] = time.time()
        
        logger.info(f"Whisper {WHISPER_MODEL_SIZE} loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
        return None


async def load_panns_model() -> Optional[object]:
    """
    Lazy-load PANNs model with memory optimization.
    Unloads after processing to free memory.
    """
    import time
    
    if model_cache["panns"] is not None:
        model_cache["panns_last_used"] = time.time()
        logger.info("Using cached PANNs model")
        return model_cache["panns"]
    
    try:
        logger.info("Loading PANNs audio tagging model (CPU)...")
        from panns_inference import AudioTagging
        
        model = AudioTagging(checkpoint_path=None, device=DEVICE)
        
        model_cache["panns"] = model
        model_cache["panns_last_used"] = time.time()
        
        logger.info("PANNs model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load PANNs: {e}")
        raise HTTPException(status_code=500, detail=f"PANNs model loading failed: {e}")


def unload_model(model_name: str):
    """Force unload a model and free memory"""
    if model_cache.get(model_name):
        logger.info(f"Unloading {model_name} model to free memory")
        model_cache[model_name] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def cleanup_old_models():
    """Background task to unload unused models"""
    import time
    
    while True:
        await asyncio.sleep(60)  # Check every minute
        current_time = time.time()
        
        for model_name in ["whisper", "panns"]:
            last_used = model_cache.get(f"{model_name}_last_used", 0)
            if current_time - last_used > MODEL_TIMEOUT and model_cache.get(model_name):
                unload_model(model_name)


# ============= FASTAPI LIFESPAN =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - minimal startup, aggressive cleanup"""
    logger.info("🚀 Starting Urban Sound Narrative API v3.0.0")
    logger.info("⚡ Memory-optimized mode: Models load on-demand")
    
    # Create audio directory
    os.makedirs("audio", exist_ok=True)
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_models())
    
    yield
    
    # Cleanup on shutdown
    cleanup_task.cancel()
    for model_name in ["whisper", "panns"]:
        unload_model(model_name)
    
    logger.info("✅ Application shutdown complete")


# ============= FASTAPI APP =============

app = FastAPI(
    title="Urban Sound Narrative API",
    description="Memory-Optimized AI Audio Processing",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ============= PROMPT BUILDERS =============

def build_groq_prompt(sound_context: str, transcript: str) -> str:
    """Optimized prompt for narrative generation"""
    return (
        f"You are a master storyteller. Based on these sounds: {sound_context}, "
        f"write a vivid 2-3 sentence urban scene.\n\n"
        f"CRITICAL: Integrate this phrase naturally: '{transcript}'\n"
        f"Use [emotion] tags for dialogue delivery (e.g., [enthusiastically]).\n"
        f"Focus on sensory details and human moments. No sound effects."
    )


def build_elevenlabs_prompt(narration_text: str, sound_context: str) -> str:
    """Optimized prompt for TTS with emotional delivery"""
    return f"""<voice>{narration_text}</voice>
<style>
Deliver with genuine emotion and dynamic pacing. For bracketed cues like [enthusiastically], 
embody the emotion fully. Vary pitch and intensity to match the scene's energy.
</style>
<context>
Urban soundscape featuring: {sound_context}. Make it cinematic and emotionally resonant.
</context>"""


# ============= CORE PROCESSING FUNCTIONS =============

async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio with Whisper, then immediately unload model.
    """
    model = await load_whisper_model()
    
    if model is None:
        logger.warning("Whisper unavailable - using fallback")
        return "(transcription unavailable)"
    
    try:
        result = model.transcribe(audio_path, fp16=False)  # fp16=False for CPU
        transcript = result["text"].strip() or "(no speech detected)"
        logger.info(f"Transcribed: {transcript}")
        return transcript
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "(transcription failed)"
        
    finally:
        # CRITICAL: Unload Whisper immediately after use
        unload_model("whisper")
        gc.collect()


async def detect_sounds(audio_path: str) -> List[str]:
    """
    Detect sounds with PANNs, then immediately unload model.
    """
    model = await load_panns_model()
    
    try:
        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != 32000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.to(DEVICE)
        
        # Inference
        from panns_inference import labels
        clipwise_output, _ = model.inference(waveform)
        clipwise_output = clipwise_output.squeeze()
        
        # Get top 5 sounds
        top_indices = clipwise_output.argsort()[-5:][::-1]
        sound_types = [labels[int(i)] for i in top_indices]
        
        logger.info(f"Detected sounds: {sound_types}")
        return sound_types
        
    finally:
        # CRITICAL: Unload PANNs immediately after use
        unload_model("panns")
        gc.collect()


async def generate_narrative(sound_types: List[str], transcript: str) -> Tuple[str, str]:
    """Generate narrative text using Groq LLM"""
    try:
        sound_context = ", ".join(sound_types)
        
        if not GROQ_API_KEY:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        
        client = Groq(api_key=GROQ_API_KEY)
        prompt = build_groq_prompt(sound_context, transcript)
        
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
        logger.error(f"GROQ API error: {e}")
        raise HTTPException(status_code=500, detail=f"Narrative generation failed: {e}")


async def synthesize_audio(narration_text: str, sound_context: str, output_filename: str) -> str:
    """Synthesize audio with ElevenLabs"""
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
        
        elevenlabs_prompt = build_elevenlabs_prompt(narration_text, sound_context)
        
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
        
        logger.info("Synthesizing speech with ElevenLabs V3")
        
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs TTS failed: {response.text}"
            )
        
        output_path = f"audio/{output_filename}"
        
        # Direct write (skip pydub if memory is tight)
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Audio saved to {output_path}")
        return output_path
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="TTS service timed out")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio synthesis failed: {e}")


# ============= STREAMING PROCESSOR =============

async def process_audio_streaming(audio_path: str, output_filename: str):
    """
    Main processing pipeline with streaming updates.
    Optimized for memory efficiency.
    """
    try:
        import json
        
        # Validate input
        if not os.path.exists(audio_path):
            yield json.dumps({"stage": "error", "message": "Upload failed", "sounds": [], "progress": 0}) + "\n"
            return
        
        yield json.dumps({"stage": "loading", "message": "🎧 Loading audio...", "sounds": [], "progress": 10}) + "\n"
        await asyncio.sleep(0.3)
        
        # STEP 1: Transcribe (Whisper loaded/unloaded here)
        yield json.dumps({"stage": "transcribing", "message": "👂 Transcribing speech...", "sounds": [], "progress": 25}) + "\n"
        transcript = await transcribe_audio(audio_path)
        
        yield json.dumps({"stage": "transcribe_complete", "message": "✅ Speech captured", "sounds": [], "progress": 40}) + "\n"
        await asyncio.sleep(0.3)
        
        # STEP 2: Detect sounds (PANNs loaded/unloaded here)
        yield json.dumps({"stage": "extracting", "message": "🔊 Analyzing sounds...", "sounds": [], "progress": 55}) + "\n"
        sound_types = await detect_sounds(audio_path)
        
        yield json.dumps({"stage": "sounds_detected", "message": "✨ Sounds identified!", "sounds": sound_types, "progress": 70}) + "\n"
        
        # Clean up input file ASAP
        try:
            Path(audio_path).unlink()
            logger.info(f"Input file deleted: {audio_path}")
        except Exception as e:
            logger.warning(f"Could not delete input: {e}")
        
        await asyncio.sleep(0.3)
        
        # STEP 3: Generate narrative (API call)
        yield json.dumps({"stage": "ai_processing", "message": "🤖 Crafting narrative...", "sounds": sound_types, "progress": 80}) + "\n"
        narration, sound_context = await generate_narrative(sound_types, transcript)
        
        # STEP 4: Synthesize voice (API call)
        yield json.dumps({"stage": "voice_generation", "message": "🎙️ Generating voice...", "sounds": sound_types, "progress": 90}) + "\n"
        output_path = await synthesize_audio(narration, sound_context, output_filename)
        
        yield json.dumps({"stage": "finalizing", "message": "✨ Finalizing...", "sounds": sound_types, "progress": 95}) + "\n"
        await asyncio.sleep(0.3)
        
        # Complete
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
        logger.error(f"Processing error: {e}", exc_info=True)
        yield json.dumps({"stage": "error", "message": f"❌ Error: {str(e)}", "sounds": [], "progress": 0}) + "\n"


# ============= API ENDPOINTS =============

@app.post("/process-audio-stream")
async def process_audio_stream(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Main endpoint for audio processing with streaming updates"""
    
    # Validate file type
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    # Generate unique filenames
    unique_id = str(uuid.uuid4())
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', file.filename)
    output_filename = f"output_{unique_id}.mp3"
    audio_path = f"audio/upload_{unique_id}_{safe_filename}"
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(audio_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Uploaded {len(content)} bytes to {audio_path}")
        
        # Schedule output cleanup (5 minutes)
        async def cleanup_output():
            await asyncio.sleep(300)
            output_path = f"audio/{output_filename}"
            if os.path.exists(output_path):
                Path(output_path).unlink()
                logger.info(f"Output cleaned: {output_path}")
        
        background_tasks.add_task(cleanup_output)
        
        # Return streaming response
        return StreamingResponse(
            process_audio_streaming(audio_path, output_filename),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        Path(audio_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    file_path = f"audio/{filename}"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/")
async def root():
    """API root with status information"""
    return {
        "status": "healthy",
        "message": "Urban Sound Narrative API v3.0.0",
        "version": "3.0.0",
        "optimizations": [
            "Lazy model loading with auto-unload",
            f"Whisper {WHISPER_MODEL_SIZE} model (minimal footprint)",
            "PANNs on-demand loading",
            "Aggressive garbage collection",
            "CPU-only inference",
            "Streaming responses"
        ],
        "memory": "Optimized for 512MB-1GB RAM environments"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_cached": model_cache["whisper"] is not None,
        "panns_cached": model_cache["panns"] is not None,
        "device": DEVICE,
        "whisper_model": WHISPER_MODEL_SIZE,
        "groq_configured": bool(GROQ_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY)
    }


# Ensure audio directory exists
os.makedirs("audio", exist_ok=True)