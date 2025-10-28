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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pydub regex warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Global model storage
models = {}

# ==================== CONFIGURATION ====================

# Optimized GROQ Configuration for Creative Narration
GROQ_MODEL_CONFIG = {
    "model": "llama-3.3-70b-versatile",  # Best free model for creative writing
    "temperature": 0.7,  # Increased for more creative, varied output
    "max_tokens": 150,  # Expanded for richer descriptions
    "top_p": 0.9,  # Nucleus sampling for quality diversity
    "frequency_penalty": 0.3,  # Reduce repetitive phrasing
    "presence_penalty": 0.2  # Encourage topic diversity
}

# Optimized ElevenLabs V3 Configuration
ELEVENLABS_V3_CONFIG = {
    "voice_id": "cgSgspJ2msm6clMCkdW9",  # Jessica voice - excellent for narration
    "model_id": "eleven_turbo_v2_5",  # Production V3 equivalent
    "voice_settings": {
        "stability": 0.4,  # Lower for more expressive variation
        "similarity_boost": 0.75,  # Balanced authenticity
        "style": 0.8,  # Higher for emotional expressiveness
        "use_speaker_boost": True
    },
    "output_format": "mp3_44100_128"  # High quality format
}

# ==================== RESPONSE MODELS ====================

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


# ==================== APP INITIALIZATION ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    logger.info("Loading AI models...")
    models['panns'] = AudioTagging(checkpoint_path=None, device='cpu')
    models['whisper'] = whisper.load_model("base", device='cpu')
    logger.info("Models loaded successfully")
    
    os.makedirs("audio", exist_ok=True)
    
    yield
    
    models.clear()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Urban Sound Narrative API",
    description="AI-Powered Audio Processing for Emotionally Rich Narration",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==================== CORE PROCESSING FUNCTIONS ====================

def build_optimized_groq_prompt(sound_context: str, transcript: str) -> str:
    """
    Build enhanced GROQ prompt for creative narrative generation.
    
    Args:
        sound_context: Comma-separated list of detected sounds
        transcript: Transcribed speech from audio
    
    Returns:
        Optimized prompt string for GROQ API
    """
    return (
        f"You are a master storyteller crafting a cinematic urban narrative. "
        f"Based on these detected sounds: {sound_context}, write a vivid, "
        f"emotionally evocative 2-3 sentence scene that immerses the reader in a bustling city moment.\n\n"
        f"CRITICAL REQUIREMENTS:\n"
        f"1. Seamlessly integrate this spoken phrase into the scene: '{transcript}'\n"
        f"2. Use square brackets to mark emotional delivery ONLY for dialogue, like: "
        f"[enthusiastically], [wearily], [sarcastically], [thoughtfully]\n"
        f"3. Make the scene feel alive - use sensory details, movement, and atmosphere\n"
        f"4. DO NOT include sound effect descriptions like (car honking) or *dog barks*\n"
        f"5. Focus on human moments and urban poetry\n\n"
        f"Example style: 'The afternoon pulse of the city beats steadily as vendors call out their wares. "
        f"A passerby mutters [sarcastically], \"Another beautiful day in paradise,\" while dodging between rushing commuters.'"
    )


def build_optimized_elevenlabs_prompt(narration_text: str, sound_context: str) -> str:
    """
    Build V3-optimized prompt for ElevenLabs with structured tags.
    
    Args:
        narration_text: Generated narrative from GROQ
        sound_context: Comma-separated detected sounds for context
    
    Returns:
        Structured prompt with voice/style/context tags
    """
    return f"""<voice>{narration_text}</voice>

<style>
Deliver this urban narrative with genuine emotion and theatrical flair. Use dynamic pacing: speak deliberately during descriptive passages to build atmosphere, then shift to natural conversational energy for any dialogue. Infuse warmth and immersion into environmental descriptions. For bracketed emotional cues like [enthusiastically] or [thoughtfully], embody that emotion fully - don't just read the word, become it. Vary your pitch and intensity to match the scene's energy. Make the listener feel present in this urban moment.
</style>

<context>
This is a vivid narration of an urban soundscape featuring: {sound_context}. The scene should feel cinematic and emotionally resonant, transporting the listener into a real city moment with all its texture and humanity.
</context>"""


async def generate_narrative_text(
    sound_types: List[str],
    transcript: str
) -> Tuple[str, str]:
    """
    Generate creative narrative text using optimized GROQ configuration.
    
    Args:
        sound_types: List of detected sound labels
        transcript: Transcribed speech
    
    Returns:
        Tuple of (narration_text, sound_context_string)
    
    Raises:
        HTTPException: On GROQ API failures
    """
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
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate narration: {str(e)}"
        )


async def synthesize_audio(
    narration_text: str,
    sound_context: str,
    output_filename: str
) -> str:
    """
    Synthesize audio using ElevenLabs V3 with optimized settings.
    
    Args:
        narration_text: Text to convert to speech
        sound_context: Sound context for V3 prompt optimization
        output_filename: Output MP3 filename
    
    Returns:
        Path to generated audio file
    
    Raises:
        HTTPException: On ElevenLabs API or processing failures
    """
    try:
        if not ELEVENLABS_API_KEY:
            raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
        
        # Build optimized V3 prompt
        elevenlabs_prompt = build_optimized_elevenlabs_prompt(narration_text, sound_context)
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_V3_CONFIG['voice_id']}"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        payload = {
            "text": elevenlabs_prompt,  # Send V3-optimized structured prompt
            "model_id": ELEVENLABS_V3_CONFIG["model_id"],
            "voice_settings": ELEVENLABS_V3_CONFIG["voice_settings"],
            "output_format": ELEVENLABS_V3_CONFIG["output_format"]
        }
        
        logger.info(f"Synthesizing speech with V3 (voice: {ELEVENLABS_V3_CONFIG['voice_id']})")
        
        tts_response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45  # Increased timeout for V3 processing
        )
        
        if tts_response.status_code != 200:
            error_detail = tts_response.text
            logger.error(f"ElevenLabs API error [{tts_response.status_code}]: {error_detail}")
            raise HTTPException(
                status_code=tts_response.status_code,
                detail=f"ElevenLabs TTS failed: {error_detail}"
            )
        
        logger.info(f"TTS synthesis successful ({len(tts_response.content)} bytes)")
        
        # Audio post-processing
        output_path = f"audio/{output_filename}"
        temp_path = f"audio/temp_{output_filename}"
        
        with open(temp_path, "wb") as f:
            f.write(tts_response.content)
        
        logger.info("Applying audio normalization and enhancement")
        
        audio = AudioSegment.from_mp3(temp_path)
        audio = audio + 6  # Gentle volume boost
        audio = audio.normalize()
        audio = audio.fade_in(100).fade_out(200)  # Professional fade effects
        
        audio.export(
            output_path,
            format="mp3",
            bitrate="192k",
            parameters=["-q:a", "0"]  # Highest quality VBR
        )
        
        Path(temp_path).unlink(missing_ok=True)
        logger.info(f"Audio exported to {output_path}")
        
        return output_path
        
    except requests.exceptions.Timeout:
        logger.error("ElevenLabs API timeout")
        raise HTTPException(
            status_code=504,
            detail="Text-to-speech service timed out. Please try again."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"ElevenLabs request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        Path(f"audio/temp_{output_filename}").unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


# ==================== STREAMING PROCESSOR ====================

async def process_audio_file_streaming(audio_path: str, output_filename: str):
    """Process audio file and yield real-time progress updates"""
    
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

        # Stage 3: Identify sounds
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

        # Stage 6: Generate narrative with optimized GROQ
        yield json.dumps({
            "stage": "ai_processing",
            "message": "🤖 Crafting cinematic narrative...",
            "sounds": sound_types,
            "progress": 70
        }) + "\n"
        await asyncio.sleep(0.5)

        narration, sound_context = await generate_narrative_text(sound_types, transcript)

        # Stage 7: Prepare for voice generation
        yield json.dumps({
            "stage": "narrating",
            "message": "✍️ Preparing emotional delivery...",
            "sounds": sound_types,
            "progress": 80
        }) + "\n"
        await asyncio.sleep(0.5)

        # Stage 8: Generate expressive voice with V3
        yield json.dumps({
            "stage": "voice_generation",
            "message": "🎙️ Generating expressive narration...",
            "sounds": sound_types,
            "progress": 90
        }) + "\n"

        output_path = await synthesize_audio(narration, sound_context, output_filename)

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
        logger.error(f"Streaming processing error: {str(e)}")
        yield json.dumps({
            "stage": "error",
            "message": f"❌ Error: {str(e)}",
            "sounds": [],
            "progress": 0
        }) + "\n"


# ==================== SYNCHRONOUS PROCESSOR ====================

def process_audio_file_sync(audio_path: str, output_filename: str) -> Tuple[str, str, List[str], str]:
    """Synchronous processing for non-streaming endpoint"""
    
    try:
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

        # Transcribe speech
        result = models['whisper'].transcribe(audio_path)
        transcript = result["text"].strip() or "(no speech)"

        # Generate narrative (run async function in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        narration, sound_context = loop.run_until_complete(
            generate_narrative_text(sound_types, transcript)
        )

        # Synthesize audio
        output_path = loop.run_until_complete(
            synthesize_audio(narration, sound_context, output_filename)
        )
        loop.close()

        return narration, output_path, sound_types, transcript

    except Exception as e:
        logger.error(f"Sync processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ==================== API ENDPOINTS ====================

@app.post("/process-audio", response_model=AudioResponse)
async def process_audio_endpoint(file: UploadFile = File(...)):
    """Process uploaded audio file - synchronous version"""
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    unique_id = str(uuid.uuid4())
    output_filename = f"output_{unique_id}.mp3"
    audio_path = f"audio/upload_{unique_id}_{file.filename}"
    
    try:
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        narration, output_path, sound_types, transcript = process_audio_file_sync(
            audio_path, output_filename
        )
        
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
        logger.error(f"Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        Path(audio_path).unlink(missing_ok=True)


@app.post("/process-audio-stream")
async def process_audio_stream(file: UploadFile = File(...)):
    """Process uploaded audio file with streaming progress updates"""
    if not file.filename.endswith((".mp3", ".wav")):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files allowed")
    
    unique_id = str(uuid.uuid4())
    output_filename = f"output_{unique_id}.mp3"
    audio_path = f"audio/upload_{unique_id}_{file.filename}"
    
    try:
        with open(audio_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return StreamingResponse(
            process_audio_file_streaming(audio_path, output_filename),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Streaming endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        await asyncio.sleep(1)
        Path(audio_path).unlink(missing_ok=True)


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve generated audio files"""
    file_path = f"audio/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Urban Sound Narrative API v2.0",
        "version": "2.0.0",
        "optimizations": [
            "GROQ Llama 3.3 70B with creative parameters",
            "ElevenLabs V3 Alpha emotional prompting",
            "Enhanced audio post-processing"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": len(models) == 2,
        "device": device,
        "groq_configured": bool(GROQ_API_KEY),
        "groq_model": GROQ_MODEL_CONFIG["model"],
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "elevenlabs_model": ELEVENLABS_V3_CONFIG["model_id"],
        "voice_id": ELEVENLABS_V3_CONFIG["voice_id"]
    }


# Ensure audio directory exists
os.makedirs("audio", exist_ok=True)