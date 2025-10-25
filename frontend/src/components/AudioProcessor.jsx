import React, { useState, useEffect, useRef } from 'react';
import { Mic, Upload, Play, Pause, Copy, Download, ArrowRight, ArrowLeft, Sparkles, Loader2, Music2, Car, FootprintsIcon as Footprints, Bell, Bus } from 'lucide-react';

function SoundNarrativeGenerator() {
  const [phase, setPhase] = useState('upload');
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [progress, setProgress] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [currentSubtitle, setCurrentSubtitle] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const fileInputRef = useRef(null);

  const processingStage = "Generating Narrative";
  
  const soundIcons = [
    { Icon: Car, label: "Vehicles" },
    { Icon: Footprints, label: "Footsteps" },
    { Icon: Bell, label: "Siren" },
    { Icon: Bus, label: "Bus" }
  ];

  const subtitles = [
    "The sharp blare of a car horn cut through the rhythmic tap-tap of footsteps on wet pavement.",
    "In the distance, a siren wailed, a lonely cry in the concrete jungle.",
    "The city breathed around them—alive, restless, never truly silent.",
    "A bus hissed to a stop nearby, its doors opening with a mechanical sigh.",
    "Conversations bubbled up and faded away, fragments of lives intersecting for mere seconds."
  ];

  const narrativeText = `The sharp blare of a car horn cut through the rhythmic tap-tap of footsteps on wet pavement. In the distance, a siren wailed, a lonely cry in the concrete jungle. The city breathed around them—alive, restless, never truly silent.

A bus hissed to a stop nearby, its doors opening with a mechanical sigh. Conversations bubbled up and faded away, fragments of lives intersecting for mere seconds before diverging again into the urban sprawl.`;

  const totalDuration = subtitles.length * 3;

  useEffect(() => {
    if (phase === 'processing') {
      const timer = setInterval(() => {
        setProgress(prev => {
          if (prev >= 100) {
            clearInterval(timer);
            setTimeout(() => setPhase('textOutput'), 600);
            return 100;
          }
          return prev + 2.5;
        });
      }, 100);
      return () => clearInterval(timer);
    }
  }, [phase]);

  useEffect(() => {
    if (phase === 'narration' && !isPaused) {
      const subtitleTimer = setInterval(() => {
        setCurrentSubtitle(prev => {
          if (prev >= subtitles.length - 1) {
            clearInterval(subtitleTimer);
            return prev;
          }
          return prev + 1;
        });
      }, 3000);

      const timeTimer = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= totalDuration) {
            clearInterval(timeTimer);
            return totalDuration;
          }
          return prev + 0.1;
        });
      }, 100);

      return () => {
        clearInterval(subtitleTimer);
        clearInterval(timeTimer);
      };
    }
  }, [phase, isPaused]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
    }
  };

  const handleGenerate = () => {
    setPhase('processing');
    setProgress(0);
  };

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(narrativeText);
    alert('Text copied to clipboard!');
  };

  const handleExport = () => {
    const blob = new Blob([narrativeText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'narrative.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

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
                Sound Narrative
              </h1>
              <p className="text-xs text-white/40 font-medium">AI-Powered Storytelling</p>
            </div>
          </div>
          <button
            onClick={() => {
              setPhase('upload');
              setFile(null);
              setFileName('');
              setCurrentSubtitle(0);
              setCurrentTime(0);
            }}
            className="group flex items-center gap-2 h-9 px-4 rounded-lg bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-200"
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
                    <span className="text-xs font-medium text-white/60 tracking-wide">Powered by Advanced AI</span>
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
                    {[...Array(window.innerWidth < 640 ? 15 : 25)].map((_, i) => (
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
                <button className="group relative flex h-20 w-20 items-center justify-center rounded-2xl bg-white/[0.08] backdrop-blur-sm border border-white/[0.1] hover:bg-white/[0.12] hover:border-white/[0.14] transition-all duration-300 hover:scale-105">
                  <Mic className="w-8 h-8 text-white/80 group-hover:text-white/95 transition-colors" strokeWidth={2} />
                </button>
                
                <div className="flex items-center gap-4">
                  <div className="h-px w-12 bg-white/[0.1]" />
                  <span className="text-xs font-semibold uppercase tracking-widest text-white/30">or</span>
                  <div className="h-px w-12 bg-white/[0.1]" />
                </div>

                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="group relative flex h-16 w-full max-w-sm items-center justify-center gap-3 rounded-2xl bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300"
                >
                  <Upload className="w-5 h-5 text-white/70 group-hover:text-white/90 transition-colors" strokeWidth={2} />
                  <span className="text-base font-medium text-white/80 group-hover:text-white/95 transition-colors">
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

              {/* Generate Button */}
              {file && (
                <div className="flex justify-center mb-14 animate-fade-in">
                  <button
                    onClick={handleGenerate}
                    className="group relative h-14 sm:h-16 rounded-2xl px-10 sm:px-14 text-base font-semibold bg-white text-black hover:bg-white/95 transition-all duration-300 hover:scale-105 overflow-hidden"
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
                        className="text-xs px-4 py-2 rounded-full bg-white/[0.05] text-white/60 border border-white/[0.08] hover:bg-white/[0.08] hover:border-white/[0.12] transition-all duration-300"
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
                        {processingStage}
                      </h2>
                      <Loader2 className="w-6 h-6 text-white/60 animate-spin-slow" strokeWidth={2} />
                    </div>
                    
                    <div className="mb-3">
                      <div className="h-1.5 w-full max-w-lg mx-auto rounded-full bg-white/[0.06] overflow-hidden">
                        <div
                          className="h-full rounded-full bg-white/80 transition-all duration-100"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                    <p className="text-sm text-white/40 font-medium">{Math.round(progress)}% Complete</p>
                  </div>

                  <div className="text-center">
                    <p className="text-sm text-white/50 mb-6 font-medium">Analyzing Audio Patterns</p>
                    <div className="flex flex-wrap gap-3 sm:gap-4 justify-center">
                      {soundIcons.map((sound, i) => {
                        const IconComponent = sound.Icon;
                        return (
                          <div
                            key={i}
                            className="flex items-center gap-2.5 sm:gap-3 px-4 sm:px-5 py-2.5 sm:py-3 rounded-2xl bg-white/[0.06] border border-white/[0.08] backdrop-blur-xl animate-scale-in"
                            style={{ animationDelay: `${i * 0.15}s` }}
                          >
                            <IconComponent className="w-5 h-5 text-white/70" strokeWidth={2} />
                            <span className="text-sm font-medium text-white/80">{sound.label}</span>
                          </div>
                        );
                      })}
                    </div>
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
                  </div>
                  
                  <div className="max-w-3xl mx-auto mb-10 bg-white/[0.03] rounded-2xl p-6 sm:p-8 border border-white/[0.06]">
                    <p className="text-base sm:text-lg leading-relaxed text-white/75 mb-6">
                      {narrativeText.split('\n\n')[0]}
                    </p>
                    <p className="text-base sm:text-lg leading-relaxed text-white/75">
                      {narrativeText.split('\n\n')[1]}
                    </p>
                  </div>
                  
                  <div className="flex flex-wrap gap-3 sm:gap-4 justify-center">
                    <button
                      onClick={() => {
                        setPhase('narration');
                        setCurrentSubtitle(0);
                        setCurrentTime(0);
                      }}
                      className="group flex items-center gap-3 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white text-black hover:bg-white/95 transition-all duration-300 hover:scale-105"
                    >
                      <Play className="w-5 h-5" strokeWidth={2.5} fill="currentColor" />
                      <span>Play Narration</span>
                      <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" strokeWidth={2.5} />
                    </button>
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-2.5 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90"
                    >
                      <Copy className="w-4.5 h-4.5" strokeWidth={2} />
                      <span>Copy</span>
                    </button>
                    <button
                      onClick={handleExport}
                      className="flex items-center gap-2.5 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90"
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
                      <span>{formatTime(totalDuration)}</span>
                    </p>
                  </div>

                  {/* Enhanced Waveform */}
                  <div className="relative mb-12">
                    <div className="absolute inset-0 bg-white/[0.02] blur-2xl" />
                    <div className="relative flex h-32 sm:h-40 items-end justify-center gap-0.5 sm:gap-1">
                      {[...Array(window.innerWidth < 640 ? 25 : 35)].map((_, i) => {
                        const totalBars = window.innerWidth < 640 ? 25 : 35;
                        const progressPercent = (currentTime / totalDuration) * 100;
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
                      {subtitles[currentSubtitle]}
                    </p>
                  </div>

                  {/* Enhanced Controls */}
                  <div className="flex flex-wrap gap-3 sm:gap-4 justify-center">
                    <button
                      onClick={() => setIsPaused(!isPaused)}
                      className="group relative flex h-14 sm:h-16 w-14 sm:w-16 items-center justify-center rounded-2xl bg-white text-black hover:bg-white/95 transition-all duration-300 hover:scale-110"
                    >
                      {isPaused ? (
                        <Play className="w-6 h-6 sm:w-7 sm:h-7" strokeWidth={2.5} fill="currentColor" />
                      ) : (
                        <Pause className="w-6 h-6 sm:w-7 sm:h-7" strokeWidth={2.5} fill="currentColor" />
                      )}
                    </button>
                    <button
                      onClick={() => {
                        setPhase('textOutput');
                        setCurrentSubtitle(0);
                        setCurrentTime(0);
                      }}
                      className="flex items-center gap-3 rounded-2xl px-6 sm:px-8 py-3 sm:py-4 text-sm sm:text-base font-semibold bg-white/[0.06] backdrop-blur-sm border border-white/[0.08] hover:bg-white/[0.1] hover:border-white/[0.12] transition-all duration-300 hover:scale-105 text-white/90"
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