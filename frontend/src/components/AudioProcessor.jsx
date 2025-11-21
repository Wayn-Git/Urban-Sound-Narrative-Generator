import React, { useState, useEffect, useRef } from 'react';
import { 
  Mic, Upload, Play, Pause, Copy, Download, ArrowRight, 
  ArrowLeft, Music2, Car, 
  FootprintsIcon as Footprints, Bell, Bus, Code, StopCircle, Check, Disc3, Zap, RotateCcw, Sparkles, FileAudio
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- UTILS ---
function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// --- API CONFIG ---
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// --- ANIMATION VARIANTS ---
const fadeInUp = {
  initial: { opacity: 0, y: 20, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
  exit: { opacity: 0, y: -20, filter: 'blur(4px)' },
  transition: { duration: 0.6, ease: [0.2, 0.8, 0.2, 1] }
};

// --- COMPONENTS ---

const SplashScreen = ({ onComplete }) => {
  return (
    <motion.div 
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[#111010]"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 1, ease: "easeInOut" } }}
    >
      <div className="relative flex flex-col items-center">
        {/* Vacuum Tube Warm-up Animation */}
        <div className="relative w-48 h-48 mb-8 flex items-center justify-center">
          {/* Filament Glow */}
          <motion.div 
            className="absolute inset-0 rounded-full bg-[#C46A29]/20 blur-3xl"
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: [0, 0.6, 0.4], scale: [0.8, 1.2, 1.1] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          />
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 2, ease: "easeIn" }}
            className="relative z-10"
          >
             <Disc3 className="w-20 h-20 text-[#D9A441] animate-[spin_4s_linear_infinite]" strokeWidth={1} />
          </motion.div>

          {/* Inner spark */}
          <motion.div 
            className="absolute w-full h-1 bg-[#D9A441]/50 blur-sm"
            initial={{ scaleX: 0, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          />
        </div>

        <motion.h1 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 1 }}
          className="text-3xl font-serif tracking-[0.15em] text-[#D9A441] uppercase"
        >
          SoundScript
        </motion.h1>
        
        <motion.div 
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: "120px", opacity: 1 }}
          transition={{ delay: 1, duration: 1.5, ease: "easeInOut" }}
          onAnimationComplete={() => setTimeout(onComplete, 600)}
          className="h-[2px] bg-gradient-to-r from-transparent via-[#C46A29] to-transparent mt-6"
        />
      </div>
    </motion.div>
  );
};

const Button = ({ children, onClick, className, variant = 'primary', disabled, icon: Icon }) => {
  const baseStyles = "relative flex items-center justify-center gap-3 px-8 py-4 rounded-2xl font-medium text-sm transition-all duration-300 overflow-hidden group backdrop-blur-sm border";
  
  const variants = {
    primary: "bg-[#C46A29] border-[#C46A29] text-[#111010] hover:bg-[#D9A441] hover:border-[#D9A441] hover:shadow-[0_0_30px_-10px_rgba(196,106,41,0.6)] font-semibold tracking-wide",
    secondary: "bg-[#1c1a1a] border-white/10 text-[#D9A441] hover:bg-[#2B1D13] hover:border-[#D9A441]/30 hover:text-[#FFEBC6]",
    danger: "bg-[#2B1D13]/50 border-red-900/30 text-red-400 hover:bg-red-900/20",
    ghost: "border-transparent text-[#8B735B] hover:text-[#EAE0D5] hover:bg-white/5"
  };

  return (
    <motion.button
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
      onClick={onClick}
      disabled={disabled}
      className={cn(baseStyles, variants[variant], disabled && "opacity-40 cursor-not-allowed grayscale", className)}
    >
      {Icon && <Icon className="w-4 h-4 relative z-10" strokeWidth={2} />}
      <span className="relative z-10">{children}</span>
    </motion.button>
  );
};

function SoundNarrativeGenerator() {
  // --- STATE ---
  const [showSplash, setShowSplash] = useState(true);
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
  const [isCopied, setIsCopied] = useState(false);
  
  const fileInputRef = useRef(null);
  const audioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

  // --- LOGIC ---
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
          if (audio.readyState < 3) await new Promise((r) => audio.addEventListener('canplay', r, { once: true }));
          await audio.play();
        } catch (err) { console.error(err); }
      };
      playAudio();
      return () => audio.removeEventListener('timeupdate', updateTime);
    }
  }, [phase, isPaused, narration]);

  useEffect(() => {
    if (isRecording) {
      recordingTimerRef.current = setInterval(() => setRecordingTime(prev => prev + 1), 1000);
    } else {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      setRecordingTime(0);
    }
    return () => { if (recordingTimerRef.current) clearInterval(recordingTimerRef.current); };
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
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
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
    } catch (err) { setError('Could not access microphone. Check permissions.'); }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const addProcessingLog = (msg) => setProcessingLogs(p => [...p, `${new Date().toLocaleTimeString()}: ${msg}`]);

  const handleCopy = () => {
    if (!narration) return;
    navigator.clipboard.writeText(narration);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleGenerate = async () => {
    if (!file) { setError('Please select a file or record audio first'); return; }
    setPhase('processing');
    setProgress(0);
    setError('');
    setDetectedSounds([]);
    setProcessingLogs([]);
    setProcessingMessage('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/process-audio-stream`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Server error (${response.status})`);

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
            if (data.message) addProcessingLog(data.message);
            if (data.sounds?.length > 0) setDetectedSounds(data.sounds);
            if (data.stage === 'complete') {
              setNarration(data.narration);
              setAudioUrl(`${API_URL}${data.audio_url}`);
              setTranscript(data.transcript);
              setTimeout(() => setPhase('textOutput'), 800);
            }
            if (data.stage === 'error') {
              setError(data.message);
              setPhase('upload');
            }
          } catch (e) { console.error(e); }
        }
      }
    } catch (err) {
      setError(err.message || 'An error occurred');
      setPhase('upload');
    }
  };

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const resetApp = () => {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime = 0; }
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
  };

  const getSoundIcon = (label) => {
    const l = label.toLowerCase();
    if (l.includes('car') || l.includes('traffic')) return Car;
    if (l.includes('foot') || l.includes('walk')) return Footprints;
    if (l.includes('bell') || l.includes('alarm')) return Bell;
    if (l.includes('bus')) return Bus;
    return Music2;
  };

  const narrativeSentences = narration ? narration.split(/[.!?]+/).filter(s => s.trim()) : [];

return (
    // Base: Off-Black with Espresso Brown undertones
    <div className="min-h-screen bg-[#111010] text-[#EAE0D5] font-sans overflow-hidden selection:bg-[#C46A29]/30 selection:text-white">
      
      <AnimatePresence mode='wait'>
        {showSplash && <SplashScreen onComplete={() => setShowSplash(false)} />}
      </AnimatePresence>

      {/* Analog Texture Background */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {/* Film Grain Texture */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.08] brightness-100 contrast-150 mix-blend-overlay"></div>
        
        {/* Warm Ambient Lighting */}
        <motion.div 
          animate={{ opacity: [0.15, 0.25, 0.15] }}
          transition={{ duration: 10, repeat: Infinity }}
          className="absolute top-[-10%] left-1/2 -translate-x-1/2 w-[80vw] h-[50vw] bg-[#C46A29]/10 rounded-full blur-[120px]" 
        />
        <div className="absolute bottom-0 left-0 w-full h-full bg-gradient-to-t from-[#2B1D13] via-[#111010]/50 to-transparent" />
      </div>

      {/* Header */}
      {!showSplash && (
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="relative z-50 px-6 py-8"
        >
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3 group cursor-default">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#2B1D13] to-[#111010] border border-white/10 flex items-center justify-center shadow-lg">
                <Zap className="w-3 h-3 text-[#D9A441] fill-current" />
              </div>
              <span className="text-sm font-serif font-semibold tracking-widest text-[#8B735B] group-hover:text-[#EAE0D5] transition-colors">SoundScript</span>
            </div>
            
            {/* FIXED START OVER BUTTON */}
            {phase !== 'upload' && (
              <motion.button
                whileHover={{ scale: 1.05, borderColor: "#D9A441" }}
                whileTap={{ scale: 0.95 }}
                onClick={resetApp}
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#D9A441]/30 bg-[#2B1D13]/50 text-[#D9A441] text-xs font-medium uppercase tracking-wider transition-colors hover:bg-[#2B1D13]"
              >
                <RotateCcw className="w-3 h-3" />
                Start Over
              </motion.button>
            )}
          </div>
        </motion.header>
      )}

      {/* Main Layout */}
      {!showSplash && (
        <main className="relative z-10 px-6 pb-20 pt-8 min-h-[80vh] flex flex-col justify-center">
          <div className="max-w-4xl mx-auto w-full">
            
            <AnimatePresence mode="wait">
              
              {/* PHASE 1: UPLOAD */}
              {phase === 'upload' && (
                <motion.div 
                  key="upload"
                  variants={fadeInUp}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  className="w-full grid md:grid-cols-5 gap-12 items-center"
                >
                  {/* Text Section */}
                  <div className="md:col-span-2 text-left">
                     <motion.div 
                      initial={{ opacity: 0 }} 
                      animate={{ opacity: 1 }} 
                      className="inline-block px-3 py-1 rounded-full border border-[#D9A441]/20 bg-[#2B1D13]/50 text-[#D9A441] text-[10px] font-bold uppercase tracking-widest mb-6"
                    >
                      Beta v3.0
                    </motion.div>
                    <motion.h1 
                      className="text-5xl md:text-6xl font-serif font-medium tracking-tight text-[#EAE0D5] mb-6 leading-[1.1]"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                      High Fidelity <br/>
                      <span className="text-[#C46A29]">Narratives.</span>
                    </motion.h1>
                    <motion.p 
                      className="text-[#8B735B] text-base font-light leading-relaxed"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 }}
                    >
                      Acoustic analysis meets generative storytelling. Upload audio to begin the signal path.
                    </motion.p>
                  </div>

                  {/* Modern Deck Interface */}
                  <div className="md:col-span-3 bg-[#181717] border border-white/5 rounded-[32px] p-2 shadow-2xl relative">
                    <div className="bg-[#111010] rounded-[24px] border border-white/5 p-8 relative overflow-hidden">
                      
                      {/* Subtle Amber Gradient Top */}
                      <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-[#C46A29]/5 to-transparent pointer-events-none" />

                      <div className="space-y-6 relative z-10">
                        {/* Record Button */}
                        <motion.button 
                          onClick={isRecording ? stopRecording : startRecording}
                          whileHover={{ scale: 1.01 }}
                          whileTap={{ scale: 0.99 }}
                          className={cn(
                            "w-full h-32 rounded-2xl border flex flex-col items-center justify-center gap-3 transition-all duration-300 group relative overflow-hidden",
                            isRecording 
                              ? "bg-[#5A1F26]/20 border-[#C46A29]/50" 
                              : "bg-[#1a1818] border-[#2B1D13] hover:border-[#D9A441]/30 hover:bg-[#222020]"
                          )}
                        >
                          {isRecording ? (
                            <>
                              <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-[#5A1F26]/40 border border-[#C46A29]/30 mb-1">
                                <div className="w-2 h-2 rounded-full bg-[#C46A29] animate-pulse" />
                                <span className="font-mono text-[10px] text-[#C46A29] tracking-widest">{formatTime(recordingTime)}</span>
                              </div>
                              <span className="text-xs text-[#C46A29] animate-pulse">Recording Input...</span>
                            </>
                          ) : (
                            <>
                              <div className="w-12 h-12 rounded-full bg-[#2B1D13] flex items-center justify-center group-hover:bg-[#3d2a1d] transition-colors">
                                <Mic className="w-5 h-5 text-[#8B735B] group-hover:text-[#D9A441]" />
                              </div>
                              <span className="text-sm font-medium text-[#8B735B] group-hover:text-[#EAE0D5]">Record Microphone</span>
                            </>
                          )}
                        </motion.button>

                        {/* UPDATED UPLOAD BUTTON (Compact Style) */}
                        <div className="relative group">
                          <input
                            ref={fileInputRef}
                            type="file"
                            accept=".mp3,.wav"
                            onChange={handleFileChange}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            disabled={isRecording}
                          />
                          <motion.div 
                            whileHover={{ scale: 1.01 }}
                            className={cn(
                              "w-full h-12 rounded-full border flex items-center justify-center gap-2 transition-all duration-300 cursor-pointer",
                              file 
                                ? "bg-[#C46A29]/10 border-[#C46A29]/40 text-[#C46A29]" 
                                : "bg-[#1a1818] border-[#2B1D13] text-[#8B735B] hover:border-[#8B735B] hover:text-[#D9A441]"
                            )}
                          >
                            {file ? (
                              <>
                                <Check className="w-4 h-4" />
                                <span className="text-xs font-medium truncate max-w-[180px]">{file.name}</span>
                              </>
                            ) : (
                              <>
                                <FileAudio className="w-4 h-4" />
                                <span className="text-xs font-medium uppercase tracking-wider">Select Audio File</span>
                              </>
                            )}
                          </motion.div>
                        </div>
                      </div>

                      {error && (
                        <motion.div 
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          className="mt-4 p-3 rounded-lg bg-[#5A1F26]/20 border border-[#5A1F26] text-red-300 text-xs text-center"
                        >
                          {error}
                        </motion.div>
                      )}

                      <div className="mt-8">
                        <Button 
                          onClick={handleGenerate} 
                          disabled={!file || isRecording} 
                          className="w-full rounded-xl h-14"
                          icon={Sparkles}
                        >
                          Process Audio
                        </Button>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* PHASE 2: PROCESSING (Animated Wheel) */}
              {phase === 'processing' && (
                <motion.div 
                  key="processing"
                  variants={fadeInUp}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  className="flex flex-col items-center justify-center"
                >
                  {/* ANIMATED LOADER */}
                  <div className="relative mb-12">
                    <div className="absolute inset-0 bg-[#D9A441]/10 blur-2xl rounded-full" />
                    <div className="relative w-40 h-40 flex items-center justify-center">
                        
                        {/* Spinning Outer Ring */}
                        <motion.div 
                          animate={{ rotate: 360 }}
                          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
                          className="absolute inset-0 rounded-full border border-dashed border-[#2B1D13]"
                        />
                        
                        {/* Pulsing Middle Ring */}
                        <motion.div 
                          animate={{ scale: [1, 1.05, 1], opacity: [0.5, 0.8, 0.5] }}
                          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                          className="absolute inset-2 rounded-full border border-[#C46A29]/20"
                        />

                        <svg className="w-full h-full -rotate-90 relative z-10">
                         {/* Progress */}
                         <motion.circle 
                          cx="80" cy="80" r="70" 
                          stroke="#C46A29" strokeWidth="3" 
                          fill="transparent" 
                          strokeDasharray="440"
                          strokeDashoffset={440 - (440 * progress) / 100}
                          strokeLinecap="round"
                          transition={{ duration: 0.5, ease: "easeOut" }}
                        />
                      </svg>
                      <span className="absolute text-4xl font-serif text-[#EAE0D5]">{Math.round(progress)}%</span>
                    </div>
                  </div>

                  <h3 className="text-xl font-serif text-[#EAE0D5] mb-2 tracking-wide">{processingMessage}</h3>
                  <p className="text-[#8B735B] text-xs mb-10 font-mono uppercase tracking-widest">Analyzing Signal</p>

                  <div className="flex flex-wrap justify-center gap-3 max-w-lg mx-auto mb-10">
                    {detectedSounds.map((sound, i) => {
                       const Icon = getSoundIcon(sound);
                       return (
                        <motion.div 
                          key={i}
                          initial={{ opacity: 0, scale: 0.8 }}
                          animate={{ opacity: 1, scale: 1 }}
                          className="flex items-center gap-2 px-4 py-2 bg-[#1a1818] border border-white/5 rounded-full"
                        >
                          <Icon className="w-3 h-3 text-[#D9A441]" />
                          <span className="text-xs text-[#EAE0D5] font-medium capitalize">{sound}</span>
                        </motion.div>
                       );
                    })}
                  </div>
                  
                  <button 
                    onClick={() => setShowProcessingDetails(!showProcessingDetails)}
                    className="text-[10px] text-[#5A4A3F] hover:text-[#D9A441] transition-colors uppercase tracking-widest flex items-center gap-2 border-b border-dashed border-[#5A4A3F] pb-1"
                  >
                    <Code className="w-3 h-3" />
                    <span>System Logs</span>
                  </button>

                  <AnimatePresence>
                    {showProcessingDetails && (
                      <motion.div 
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 200 }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-6 w-full max-w-lg bg-[#0a0909] border border-[#2B1D13] rounded-xl p-4 overflow-y-auto font-mono text-[10px] text-[#8B735B]"
                      >
                        {processingLogs.map((log, i) => (
                          <div key={i} className="mb-1 opacity-70 border-l border-[#5A1F26] pl-2">{log}</div>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}

              {/* PHASE 3: OUTPUT */}
              {phase === 'textOutput' && (
                <motion.div 
                  key="output"
                  variants={fadeInUp}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                >
                  <div className="mb-6 flex items-center justify-between">
                    <div>
                      <h2 className="text-2xl font-serif text-[#EAE0D5]">Narrative Output</h2>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="secondary" onClick={handleCopy} className="px-4 h-10 rounded-xl">
                        {isCopied ? <Check className="w-4 h-4 text-green-400"/> : <Copy className="w-4 h-4"/>}
                      </Button>
                      <Button 
                        variant="secondary" 
                        className="px-4 h-10 rounded-xl"
                        onClick={() => {
                          const blob = new Blob([narration], { type: 'text/plain' });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement('a'); a.href = url; a.download = 'narrative.txt'; a.click();
                        }} 
                      ><Download className="w-4 h-4"/></Button>
                    </div>
                  </div>

                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mb-10 bg-[#181717] p-10 rounded-3xl border border-white/5 shadow-xl"
                  >
                    <p className="font-serif text-xl md:text-2xl leading-loose text-[#C9CCD1] antialiased tracking-wide whitespace-pre-wrap">
                      {narration}
                    </p>
                  </motion.div>

                  <div className="flex justify-center">
                    <Button 
                      onClick={() => {
                        setPhase('narration');
                        setCurrentSubtitle(0);
                        setCurrentTime(0);
                        setIsPaused(false);
                      }}
                      icon={Play}
                      className="px-12 py-4 text-base rounded-full shadow-[0_10px_30px_-5px_rgba(196,106,41,0.4)]"
                    >
                      Start Playback
                    </Button>
                  </div>
                </motion.div>
              )}

              {/* PHASE 4: PLAYBACK */}
              {phase === 'narration' && (
                <motion.div 
                  key="narration"
                  variants={fadeInUp}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  className="flex flex-col items-center"
                >
                  {/* Sleek Modern Player */}
                  <div className="w-full max-w-2xl bg-[#181717] border border-white/5 rounded-[40px] p-8 md:p-12 shadow-2xl relative overflow-hidden">
                    {/* Top Status */}
                    <div className="flex items-center justify-between mb-10">
                         <div className="flex items-center gap-2">
                             <div className="w-2 h-2 rounded-full bg-[#C46A29] animate-pulse" />
                             <span className="text-[10px] font-bold text-[#8B735B] uppercase tracking-widest">Now Playing</span>
                         </div>
                         <span className="text-[10px] font-mono text-[#5A4A3F]">{formatTime(currentTime)} / {formatTime(audioRef.current?.duration || 0)}</span>
                    </div>

                    {/* Clean Gold Waveform */}
                    <div className="h-32 flex items-center justify-center gap-1.5 mb-12">
                      {[...Array(32)].map((_, i) => {
                         const isActive = (i / 32) * (audioRef.current?.duration || 1) < currentTime;
                         return (
                          <motion.div 
                            key={i} 
                            animate={{ 
                              height: isActive && !isPaused ? [24, 80, 24] : 24,
                              opacity: isActive ? 1 : 0.3,
                              backgroundColor: isActive ? "#C46A29" : "#2B1D13",
                            }}
                            transition={{ 
                              duration: 0.8, 
                              repeat: Infinity, 
                              delay: i * 0.04,
                              ease: "easeInOut"
                            }}
                            className="w-1.5 rounded-full"
                          />
                         )
                      })}
                    </div>

                    <div className="min-h-[80px] flex items-center justify-center text-center relative z-10">
                      <AnimatePresence mode='wait'>
                        <motion.p 
                          key={currentSubtitle}
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -5 }}
                          className="font-serif text-xl text-[#EAE0D5] leading-relaxed"
                        >
                          {narrativeSentences[currentSubtitle]}
                        </motion.p>
                      </AnimatePresence>
                    </div>
                  </div>

                  <audio ref={audioRef} src={audioUrl} crossOrigin="anonymous" onEnded={() => setIsPaused(true)} className="hidden" />

                  <div className="flex items-center gap-6 mt-8 bg-[#111010]/80 backdrop-blur-md p-2 rounded-full border border-white/5">
                    <button onClick={() => setPhase('textOutput')} className="w-12 h-12 rounded-full flex items-center justify-center hover:bg-white/5 text-[#8B735B] transition-colors"><ArrowLeft className="w-5 h-5" /></button>
                    
                    <motion.button 
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        if (audioRef.current) isPaused ? audioRef.current.play() : audioRef.current.pause();
                        setIsPaused(!isPaused);
                      }}
                      className="w-16 h-16 flex items-center justify-center bg-[#C46A29] text-[#111010] rounded-full shadow-lg"
                    >
                      {isPaused ? <Play className="w-6 h-6 fill-current ml-1" /> : <Pause className="w-6 h-6 fill-current" />}
                    </motion.button>

                    <button 
                      onClick={() => { const a = document.createElement('a'); a.href = audioUrl; a.download = 'narration.mp3'; a.click(); }} 
                      className="w-12 h-12 rounded-full flex items-center justify-center hover:bg-white/5 text-[#8B735B] transition-colors"
                    >
                      <Download className="w-5 h-5" />
                    </button>
                  </div>
                </motion.div>
              )}

            </AnimatePresence>
          </div>
        </main>
      )}
    </div>
  );
}

export default SoundNarrativeGenerator;