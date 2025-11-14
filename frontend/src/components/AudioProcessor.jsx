import React, { useState, useEffect, useRef } from 'react';
import { Mic, Upload, Play, Pause, Copy, Download, ArrowRight, ArrowLeft, Sparkles, Loader2, Music2, Car, FootprintsIcon as Footprints, Bell, Bus, ChevronDown, ChevronUp, Code, StopCircle } from 'lucide-react';

// IMPORTANT: Update this URL with your ngrok URL from Colab
// Copy the URL that looks like: https://xxxxxxxxxxxx.ngrok-free.app
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// const API_URL = import.meta.env.VITE_API_URL;

console.log("API URL:", API_URL);


function SoundNarrativeGenerator() {
  const [phase, setPhase] = useState('upload');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [progress, setProgress] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [currentSubtitle, setCurrentSubtitle] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [narration, setNarration] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [detectedSounds, setDetectedSounds] = useState([]);
  const [transcript, setTranscript] = useState('');
  const [processingMessage, setProcessingMessage] = useState('');
  const [error, setError] = useState('');
  const [showProcessingDetails, setShowProcessingDetails] = useState(false);
  const [processingLogs, setProcessingLogs] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  const fileInputRef = useRef(null);
  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

  useEffect(() => {
    if (phase === 'narration' && !isPaused && audioRef.current) {
      const audio = audioRef.current;
      
      const updateTime = () => {
        setCurrentTime(audio.currentTime);
        const sentences = narration.split(/[.!?]+/).filter(s => s.trim());
        if (sentences.length > 0 && audio.duration) {
          const timePerSentence = audio.duration / sentences.length;
          const subtitleIndex = Math.floor(audio.currentTime / timePerSentence);
          setCurrentSubtitle(Math.min(subtitleIndex, sentences.length - 1));
        }
      };

      audio.addEventListener('timeupdate', updateTime);
      const playAudio = async () => {
  try {
    console.log('Audio URL:', audio.src);
    console.log('Audio ready state:', audio.readyState);
    
    // Wait for audio to be ready
    if (audio.readyState < 3) {
      console.log('Waiting for audio to load...');
      await new Promise((resolve) => {
        audio.addEventListener('canplay', resolve, { once: true });
      });
    }
    
    await audio.play();
    console.log('Audio playing successfully');
  } catch (err) {
    console.error('Audio play error:', err);
    console.error('Error details:', {
      name: err.name,
      message: err.message,
      code: err.code
    });
  }
};

playAudio();

      return () => {
        audio.removeEventListener('timeupdate', updateTime);
      };
    }
  }, [phase, isPaused, narration]);

  useEffect(() => {
    if (isRecording) {
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      setRecordingTime(0);
    }
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, [isRecording]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.mp3') && !selectedFile.name.endsWith('.wav')) {
        setError('Please select an MP3 or WAV file');
        return;
      }
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

       const recordedFile = new File([audioBlob], `recording_${Date.now()}.webm`, { type: 'audio/webm' });

        setFile(recordedFile);
        setFileName(recordedFile.name);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError('');
    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const addProcessingLog = (message) => {
    setProcessingLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const handleGenerate = async () => {
    if (!file) {
      setError('Please select a file or record audio first');
      return;
    }

    setPhase('processing');
    setProgress(0);
    setError('');
    setDetectedSounds([]);
    setProcessingLogs([]);
    setProcessingMessage('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      addProcessingLog('Initializing audio processing pipeline...');
      const response = await fetch(`${API_URL}/process-audio-stream`, {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error (${response.status}): ${errorText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            setProgress(data.progress || 0);
            setProcessingMessage(data.message || '');
            
            if (data.stage === 'loading') {
              addProcessingLog('Loading audio file and validating format...');
              addProcessingLog('Resampling audio to 32kHz mono channel...');
            } else if (data.stage === 'extracting') {
              addProcessingLog('Running PANNs audio tagging model...');
              addProcessingLog('Extracting acoustic features from waveform...');
            } else if (data.stage === 'identifying') {
              addProcessingLog('Analyzing frequency patterns...');
              addProcessingLog('Classifying sound events...');
            } else if (data.stage === 'sounds_detected') {
              addProcessingLog(`Detected ${data.sounds.length} distinct sound classes`);
              data.sounds.forEach(sound => addProcessingLog(`  → ${sound}`));
            } else if (data.stage === 'transcribing') {
              addProcessingLog('Loading Whisper speech recognition model...');
              addProcessingLog('Transcribing audio content...');
            } else if (data.stage === 'ai_processing') {
              addProcessingLog('Connecting to Groq LLM API...');
              addProcessingLog('Generating narrative with Llama 3.3 70B...');
            } else if (data.stage === 'voice_generation') {
              addProcessingLog('Connecting to EOpen Source TTS API...');
              addProcessingLog('Synthesizing voice narration...');
            } else if (data.stage === 'finalizing') {
              addProcessingLog('Normalizing audio levels...');
              addProcessingLog('Encoding to MP3 format...');
            }
            
            if (data.sounds && data.sounds.length > 0) {
              setDetectedSounds(data.sounds);
            }
            
            if (data.stage === 'complete') {
              setNarration(data.narration);
              setAudioUrl(`${API_URL}${data.audio_url}`);
              setTranscript(data.transcript);
              addProcessingLog('Processing complete! Audio ready for playback.');
              setTimeout(() => setPhase('textOutput'), 600);
            }
            
            if (data.stage === 'error') {
              setError(data.message);
              addProcessingLog(`ERROR: ${data.message}`);
              setPhase('upload');
            }
          } catch (e) {
            console.error('Error parsing JSON:', e, line);
          }
        }
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'An error occurred while processing the audio');
      addProcessingLog(`FATAL ERROR: ${err.message}`);
      setPhase('upload');
    }
  };

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const formatRecordingTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const handleCopy = (e) => {
    navigator.clipboard.writeText(narration);
    const button = e.target.closest('button');
    const span = button.querySelector('span');
    if (span) {
      const originalText = span.textContent;
      span.textContent = 'Copied!';
      setTimeout(() => {
        span.textContent = originalText;
      }, 2000);
    }
  };

  const handleExport = () => {
    const blob = new Blob([narration], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'narrative.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadAudio = () => {
    if (audioUrl) {
      const a = document.createElement('a');
      a.href = audioUrl;
      a.download = 'narration.mp3';
      a.click();
    }
  };

  const resetApp = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setPhase('upload');
    setFile(null);
    setFileName('');
    setCurrentSubtitle(0);
    setCurrentTime(0);
    setError('');
    setNarration('');
    setAudioUrl('');
    setDetectedSounds([]);
    setProcessingLogs([]);
    setTranscript('');
    setProgress(0);
    setProcessingMessage('');
    setShowProcessingDetails(false);
  };

  const getSoundIcon = (soundLabel) => {
    const lowerLabel = soundLabel.toLowerCase();
    if (lowerLabel.includes('car') || lowerLabel.includes('vehicle') || lowerLabel.includes('traffic')) {
      return Car;
    } else if (lowerLabel.includes('foot') || lowerLabel.includes('walk')) {
      return Footprints;
    } else if (lowerLabel.includes('bell') || lowerLabel.includes('siren') || lowerLabel.includes('alarm')) {
      return Bell;
    } else if (lowerLabel.includes('bus')) {
      return Bus;
    }
    return Music2;
  };

  const narrativeSentences = narration ? narration.split(/[.!?]+/).filter(s => s.trim()) : [];

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}>
      {/* Blur Gradient Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[10%] left-[15%] w-[500px] h-[500px] bg-white/[0.03] rounded-full blur-[120px] animate-float" />
        <div className="absolute bottom-[15%] right-[10%] w-[600px] h-[600px] bg-white/[0.02] rounded-full blur-[100px] animate-float-delayed" />
        <div className="absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-white/[0.015] rounded-full blur-[140px] animate-pulse-slow" />
      </div>

      {/* Header */}
      <header className="relative z-10 px-4 sm:px-6 lg:px-8 py-6 border-b border-white/[0.06]">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-white/[0.08] backdrop-blur-sm flex items-center justify-center border border-white/[0.06]">
              <Music2 className="w-5 h-5 text-white/90" strokeWidth={2} />
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-white/95">
                Sound Script
              </h1>
              <p className="text-xs text-white/40 font-medium">AI-Powered Storytelling</p>
            </div>
          </div>
          <button
            onClick={resetApp}
            className="group flex items-center gap-2 h-9 px-4 rounded-lg bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-200 cursor-pointer"
          >
            <Sparkles className="w-4 h-4 text-white/70 group-hover:text-white/90 transition-colors" strokeWidth={2} />
            <span className="text-sm font-medium text-white/80 group-hover:text-white/95 transition-colors">New Project</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-20">
        <div className="max-w-5xl mx-auto">
          
          {/* UPLOAD PHASE */}
          {phase === 'upload' && (
            <div className="animate-fade-in">
              <div className="text-center mb-12 sm:mb-16">
                <div className="inline-block mb-6">
                  <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/[0.04] backdrop-blur-sm border border-white/[0.06]">
                    <div className="w-1.5 h-1.5 rounded-full bg-white/60 animate-pulse" />
                    <span className="text-xs font-medium text-white/60 tracking-wide">Powered By Not Eleven Labs & Groq</span>
                  </div>
                </div>
                <h2 className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold tracking-tight mb-6 text-white/95 leading-[1.1]">
                  Transform Sound<br/>Into Story
                </h2>
                <p className="text-lg sm:text-xl text-white/50 mb-4 max-w-2xl mx-auto font-medium">
                  Upload any audio and watch AI craft immersive narratives
                </p>
                <p className="text-sm text-white/30 max-w-xl mx-auto">
                  From bustling streets to tranquil nature—every sound becomes a vivid tale
                </p>
              </div>

              {/* Waveform Visualizer Card */}
              <div className="group relative mb-10 sm:mb-12">
                <div className="absolute inset-0 bg-white/[0.02] rounded-3xl blur-xl group-hover:blur-2xl transition-all duration-500" />
                <div className="relative bg-white/[0.03] backdrop-blur-xl border border-white/[0.08] rounded-3xl p-8 sm:p-12 overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                  <div className="relative flex h-48 sm:h-56 items-center justify-center gap-1.5 sm:gap-2">
                    {[...Array(25)].map((_, i) => (
                      <div
                        key={i}
                        className="w-1 sm:w-1.5 rounded-full bg-white/70 animate-wave"
                        style={{
                          height: `${Math.random() * 60 + 20}%`,
                          animationDelay: `${i * 0.08 - 1}s`,
                          animationDuration: '2s'
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Upload Controls */}
              <div className="mb-10 sm:mb-12 flex flex-col items-center gap-6 sm:flex-row sm:justify-center">
                {!isRecording ? (
                  <button 
                    onClick={startRecording}
                    className="group relative flex h-20 w-20 items-center justify-center rounded-2xl bg-white/[0.08] backdrop-blur-sm border border-white/[0.1] hover:bg-white/[0.12] hover:border-white/[0.14] transition-all duration-300 hover:scale-105 cursor-pointer"
                  >
                    <Mic className="w-8 h-8 text-white/80 group-hover:text-white/95 transition-colors" strokeWidth={2} />
                  </button>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <button 
                      onClick={stopRecording}
                      className="group relative flex h-20 w-20 items-center justify-center rounded-2xl bg-red-500/20 backdrop-blur-sm border border-red-500/30 hover:bg-red-500/30 hover:border-red-500/40 transition-all duration-300 animate-pulse cursor-pointer"
                    >
                      <StopCircle className="w-8 h-8 text-red-400" strokeWidth={2} fill="currentColor" />
                    </button>
                    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-red-500/10 border border-red-500/20">
                      <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
                      <span className="text-sm font-mono text-red-300">{formatRecordingTime(recordingTime)}</span>
                    </div>
                  </div>
                )}
                
                <div className="flex items-center gap-4">
                  <div className="h-px w-12 bg-white/[0.1]" />
                  <span className="text-xs font-semibold uppercase tracking-widest text-white/30">or</span>
                  <div className="h-px w-12 bg-white/[0.1]" />
                </div>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isRecording}
                  className="group relative flex h-16 w-full max-w-sm items-center justify-center gap-3 rounded-2xl bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  <Upload className="w-5 h-5 text-white/70 group-hover:text-white/90 transition-colors" strokeWidth={2} />
                  <span className="text-base font-medium text-white/80 group-hover:text-white/95 transition-colors truncate px-2">
                    {fileName || 'Choose Audio File'}
                  </span>
                  {fileName && <div className="w-2 h-2 rounded-full bg-white/60" />}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp3,.wav"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>

              {/* Error Display */}
              {error && (
                <div className="mb-8 max-w-2xl mx-auto animate-fade-in">
                  <div className="bg-red-500/10 border border-red-500/20 rounded-2xl p-4 text-center">
                    <p className="text-red-400 text-sm font-medium">{error}</p>
                  </div>
                </div>
              )}

              {/* Generate Button */}
              {file && !error && !isRecording && (
                <div className="flex justify-center mb-14 animate-fade-in">
                  <button
                    onClick={handleGenerate}
                    className="group relative h-14 sm:h-16 rounded-2xl px-10 sm:px-14 text-base font-semibold bg-white text-black hover:bg-white/95 transition-all duration-300 hover:scale-105 overflow-hidden cursor-pointer"
                  >
                    <span className="relative z-10 flex items-center gap-3">
                      <span>Generate Narrative</span>
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" strokeWidth={2.5} />
                    </span>
                  </button>
                </div>
              )}

              {/* Example Preview */}
              <div className="group relative">
                <div className="absolute inset-0 bg-white/[0.01] rounded-3xl blur-xl" />
                <div className="relative bg-white/[0.03] backdrop-blur-xl border border-white/[0.06] rounded-3xl p-6 sm:p-8">
                  <div className="flex items-center gap-2 mb-5">
                    <div className="w-1.5 h-1.5 rounded-full bg-white/50" />
                    <h3 className="text-xs font-bold uppercase tracking-widest text-white/50">Example Output</h3>
                  </div>
                  <div className="flex flex-wrap gap-2 mb-5">
                    {['Cars passing', 'Dog barking', 'Children laughing', 'Construction', 'Bicycle bell'].map((tag) => (
                      <span
                        key={tag}
                        className="text-xs px-4 py-2 rounded-full bg-white/[0.05] text-white/60 border border-white/[0.08] hover:bg-white/[0.08] hover:border-white/[0.12] transition-all duration-300 cursor-default"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <p className="text-sm leading-relaxed text-white/60 italic">
                    "The afternoon pulse of the city beats steadily. Cars rush past in waves, their engines humming a familiar urban melody. Somewhere nearby, a dog barks—sharp, insistent..."
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* PROCESSING PHASE */}
          {phase === 'processing' && (
            <div className="animate-fade-in">
              <div className="relative">
                <div className="absolute inset-0 bg-white/[0.02] rounded-3xl blur-2xl animate-pulse-slow" />
                <div className="relative bg-white/[0.04] backdrop-blur-xl border border-white/[0.08] rounded-3xl p-10 sm:p-14 overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/[0.03] to-transparent -translate-x-full animate-shimmer-fast" />
                  
                  <div className="relative text-center mb-10">
                    <div className="inline-flex items-center gap-3 mb-6">
                      <div className="relative">
                        <div className="w-2.5 h-2.5 rounded-full bg-white/60 animate-ping absolute" />
                        <div className="w-2.5 h-2.5 rounded-full bg-white/80 relative" />
                      </div>
                      <h2 className="text-2xl font-bold text-white/90">
                        Generating Narrative
                      </h2>
                      <Loader2 className="w-6 h-6 text-white/60 animate-spin-slow" strokeWidth={2} />
                    </div>
                    
                    <div className="mb-3">
                      <div className="h-1.5 w-full max-w-lg mx-auto rounded-full bg-white/[0.06] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-white/80 transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                    <p className="text-sm text-white/40 font-medium mb-2">{Math.round(progress)}% Complete</p>
                    <p className="text-sm text-white/60 font-medium animate-pulse">{processingMessage}</p>
                  </div>

                  <div className="text-center">
                    <p className="text-sm text-white/50 mb-6 font-medium">
                      {detectedSounds.length > 0 ? 'Detected Sounds' : 'Analyzing Audio Patterns'}
                    </p>
                    {detectedSounds.length > 0 ? (
                      <div className="flex flex-wrap gap-3 sm:gap-4 justify-center mb-6">
                        {detectedSounds.map((sound, i) => {
                          const IconComponent = getSoundIcon(sound);
                          return (
                            <div
                              key={i}
                              className="flex items-center gap-2.5 sm:gap-3 px-4 sm:px-5 py-2.5 sm:py-3 rounded-2xl bg-white/[0.06] border border-white/[0.08] backdrop-blur-xl animate-scale-in"
                              style={{ animationDelay: `${i * 0.1}s` }}
                            >
                              <IconComponent className="w-5 h-5 text-white/70" strokeWidth={2} />
                              <span className="text-sm font-medium text-white/80">{sound}</span>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center gap-2 mb-6">
                        <div className="w-2 h-2 rounded-full bg-white/40 animate-pulse" style={{ animationDelay: '0s' }} />
                        <div className="w-2 h-2 rounded-full bg-white/40 animate-pulse" style={{ animationDelay: '0.2s' }} />
                        <div className="w-2 h-2 rounded-full bg-white/40 animate-pulse" style={{ animationDelay: '0.4s' }} />
                      </div>
                    )}

                    {/* Processing Details Toggle */}
                    {processingLogs.length > 0 && (
                      <div className="mt-8">
                        <button
                          onClick={() => setShowProcessingDetails(!showProcessingDetails)}
                          className="group flex items-center gap-2 mx-auto px-4 py-2 rounded-lg bg-white/[0.04] border border-white/[0.06] hover:bg-white/[0.06] hover:border-white/[0.08] transition-all duration-200 cursor-pointer"
                        >
                          <Code className="w-4 h-4 text-white/60" strokeWidth={2} />
                          <span className="text-xs font-medium text-white/60">
                            {showProcessingDetails ? 'Hide' : 'View'} Processing Details
                          </span>
                          {showProcessingDetails ? (
                            <ChevronUp className="w-4 h-4 text-white/60" strokeWidth={2} />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-white/60" strokeWidth={2} />
                          )}
                        </button>

                        {showProcessingDetails && (
                          <div className="mt-4 max-h-64 overflow-y-auto bg-black/40 backdrop-blur-sm border border-white/[0.06] rounded-xl p-4 text-left animate-fade-in">
                            <div className="font-mono text-xs space-y-1">
                              {processingLogs.map((log, i) => (
                                <div key={i} className="text-white/50 leading-relaxed">
                                  {log}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TEXT OUTPUT PHASE */}
          {phase === 'textOutput' && (
            <div className="animate-fade-in">
              <div className="relative">
                <div className="absolute inset-0 bg-white/[0.01] rounded-3xl blur-xl" />
                <div className="relative bg-white/[0.04] backdrop-blur-xl border border-white/[0.08] rounded-3xl p-8 sm:p-12">
                  <div className="text-center mb-8">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/[0.06] border border-white/[0.08] mb-4">
                      <div className="w-1.5 h-1.5 rounded-full bg-white/70" />
                      <span className="text-xs font-semibold text-white/70">Generation Complete</span>
                    </div>
                    <h2 className="text-2xl sm:text-3xl font-bold mb-2 text-white/95">Your Narrative</h2>
                    {transcript && transcript !== '(no speech)' && (
                      <p className="text-sm text-white/40 italic mt-2">Transcript: "{transcript}"</p>
                    )}
                  </div>
                  
                  <div className="max-w-3xl mx-auto mb-10 bg-white/[0.03] rounded-2xl p-6 sm:p-8 border border-white/[0.06]">
                    <p className="text-base sm:text-lg leading-relaxed text-white/75">
                      {narration}
                    </p>
                  </div>
                  
                  <div className="flex flex-wrap gap-3 sm:gap-4 justify-center">
                    <button
                      onClick={() => {
                        setPhase('narration');
                        setCurrentSubtitle(0);
                        setCurrentTime(0);
                        setIsPaused(false);
                      }}
                      className="group flex items-center gap-3 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white text-black hover:bg-white/95 transition-all duration-300 hover:scale-105 cursor-pointer"
                    >
                      <Play className="w-5 h-5" strokeWidth={2.5} fill="currentColor" />
                      <span>Play Narration</span>
                      <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" strokeWidth={2.5} />
                    </button>
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-2.5 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90 cursor-pointer"
                    >
                      <Copy className="w-4.5 h-4.5" strokeWidth={2} />
                      <span>Copy</span>
                    </button>
                    <button
                      onClick={handleExport}
                      className="flex items-center gap-2.5 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90 cursor-pointer"
                    >
                      <Download className="w-4.5 h-4.5" strokeWidth={2} />
                      <span>Export</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* NARRATION PHASE */}
          {phase === 'narration' && (
            <div className="animate-fade-in">
              <div className="relative">
                <div className="absolute inset-0 bg-white/[0.02] rounded-3xl blur-2xl animate-pulse-slow" />
                <div className="relative bg-white/[0.04] backdrop-blur-xl border border-white/[0.08] rounded-3xl p-8 sm:p-12">
                  <div className="text-center mb-10">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/[0.06] border border-white/[0.08] mb-4">
                      <div className="w-1.5 h-1.5 rounded-full bg-white/70 animate-pulse" />
                      <span className="text-xs font-semibold text-white/70">Now Playing</span>
                    </div>
                    <h2 className="text-2xl font-bold mb-3 text-white/95">Audio Narration</h2>
                    <p className="text-base text-white/50 font-mono">
                      <span className="text-white/70">{formatTime(currentTime)}</span>
                      <span className="mx-2 text-white/30">/</span>
                      <span>{formatTime(audioRef.current?.duration || 0)}</span>
                    </p>
                  </div>

                  {/* Enhanced Waveform */}
                  <div className="relative mb-12">
                    <div className="absolute inset-0 bg-white/[0.02] blur-2xl" />
                    <div className="relative flex h-32 sm:h-40 items-end justify-center gap-0.5 sm:gap-1">
                      {[...Array(35)].map((_, i) => {
                        const totalBars = 35;
                        const audioDuration = audioRef.current?.duration || 1;
                        const progressPercent = (currentTime / audioDuration) * 100;
                        const barPosition = (i / totalBars) * 100;
                        const isPastProgress = barPosition <= progressPercent;
                        const randomHeight = `${Math.random() * 60 + 20}%`;
                        
                        return (
                          <div
                            key={i}
                            className={`w-0.5 sm:w-1 rounded-full transition-all duration-300 ${
                              isPastProgress && !isPaused ? 'animate-wave bg-white' : 'bg-white/30'
                            }`}
                            style={{
                              height: isPastProgress ? randomHeight : '20%',
                              animationDelay: isPastProgress && !isPaused ? `${i * 0.08 - 2}s` : undefined,
                              animationDuration: isPastProgress && !isPaused ? '1.8s' : undefined
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Subtitle Display */}
                  <div className="min-h-24 sm:min-h-28 flex items-center justify-center mb-10 px-4">
                    <p className="text-lg sm:text-xl lg:text-2xl leading-relaxed text-center max-w-3xl font-medium text-white/85 animate-pulse-subtle">
                      {narrativeSentences[currentSubtitle] || narration}
                    </p>
                  </div>

                  {/* Hidden Audio Element */}
<audio
  ref={audioRef}
  src={audioUrl}
  crossOrigin="anonymous"  // ADD THIS
  onEnded={() => {
    setIsPaused(true);
    setCurrentTime(0);
    setCurrentSubtitle(0);
  }}
  className="hidden"
/>

                  {/* Enhanced Controls */}
                  <div className="flex flex-wrap gap-3 sm:gap-4 justify-center">
                    <button
                      onClick={() => {
                        if (audioRef.current) {
                          if (isPaused) {
                            audioRef.current.play();
                          } else {
                            audioRef.current.pause();
                          }
                          setIsPaused(!isPaused);
                        }
                      }}
                      className="group relative flex h-14 sm:h-16 w-14 sm:w-16 items-center justify-center rounded-2xl bg-white text-black hover:bg-white/95 transition-all duration-300 hover:scale-110 cursor-pointer"
                    >
                      {isPaused ? (
                        <Play className="w-6 h-6 sm:w-7 sm:h-7" strokeWidth={2.5} fill="currentColor" />
                      ) : (
                        <Pause className="w-6 h-6 sm:w-7 sm:h-7" strokeWidth={2.5} fill="currentColor" />
                      )}
                    </button>
                    <button
                      onClick={handleDownloadAudio}
                      className="flex items-center gap-2.5 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90 cursor-pointer"
                    >
                      <Download className="w-4.5 h-4.5" strokeWidth={2} />
                      <span>Download Audio</span>
                    </button>
                    <button
                      onClick={() => {
                        if (audioRef.current) {
                          audioRef.current.pause();
                          audioRef.current.currentTime = 0;
                        }
                        setPhase('textOutput');
                        setCurrentSubtitle(0);
                        setCurrentTime(0);
                        setIsPaused(false);
                      }}
                      className="flex items-center gap-3 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90 cursor-pointer"
                    >
                      <ArrowLeft className="w-5 h-5" strokeWidth={2} />
                      <span>Back to Text</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <style>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes wave {
          0%, 100% { transform: scaleY(0.3); }
          50% { transform: scaleY(1); }
        }
        @keyframes shimmer-fast {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
        @keyframes float {
          0%, 100% { transform: translate(0, 0); }
          33% { transform: translate(30px, -30px); }
          66% { transform: translate(-20px, 20px); }
        }
        @keyframes float-delayed {
          0%, 100% { transform: translate(0, 0); }
          33% { transform: translate(-30px, 30px); }
          66% { transform: translate(20px, -20px); }
        }
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 0.7; }
        }
        @keyframes pulse-subtle {
          0%, 100% { opacity: 0.85; }
          50% { opacity: 1; }
        }
        @keyframes scale-in {
          from { opacity: 0; transform: scale(0.9); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-fade-in {
          animation: fade-in 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .animate-wave {
          animation: wave 1.8s infinite ease-in-out;
        }
        .animate-shimmer-fast {
          animation: shimmer-fast 2.5s infinite;
        }
        .animate-float {
          animation: float 20s infinite ease-in-out;
        }
        .animate-float-delayed {
          animation: float-delayed 25s infinite ease-in-out;
        }
        .animate-pulse-slow {
          animation: pulse-slow 4s infinite ease-in-out;
        }
        .animate-pulse-subtle {
          animation: pulse-subtle 3s infinite ease-in-out;
        }
        .animate-scale-in {
          animation: scale-in 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          opacity: 0;
        }
        .animate-spin-slow {
          animation: spin-slow 3s linear infinite;
        }
      `}</style>
    </div>
  );
}

export default SoundNarrativeGenerator;