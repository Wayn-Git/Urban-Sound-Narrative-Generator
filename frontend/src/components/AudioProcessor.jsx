import React, { useState, useEffect } from 'react';
import axios from 'axios';


const API_URL = "https://d2b1131ccd5b.ngrok-free.app"; // Update this!

function AudioProcessor() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [detectedSounds, setDetectedSounds] = useState([]);
  const [narration, setNarration] = useState('');
  const [audioUrl, setAudioUrl] = useState('');
  const [error, setError] = useState('');
  const [fileName, setFileName] = useState('');

  // Processing stages with fun messages
  const processingStages = [
    { message: "🎧 Analyzing your audio...", duration: 2000 },
    { message: "🔊 Extracting sound patterns...", duration: 2500 },
    { message: "🎵 Identifying urban soundscapes...", duration: 2000 },
    { message: "🤖 Feeding sounds to AI brain...", duration: 2500 },
    { message: "✍️ Crafting your narrative...", duration: 3000 },
    { message: "🎙️ Generating voice narration...", duration: 3000 },
    { message: "✨ Adding final touches...", duration: 1500 }
  ];

  const possibleSounds = [
    "Traffic hum", "Footsteps", "Conversation murmurs", "Car horns",
    "Bicycle bells", "Street vendor calls", "Birds chirping", "Wind rustling",
    "Music in the distance", "Construction sounds", "Children playing",
    "Cafe ambiance", "Sirens", "Door creaks", "Rain pattering"
  ];

  useEffect(() => {
    if (loading) {
      let currentStage = 0;
      
      // Simulate sound detection
      const soundInterval = setInterval(() => {
        if (detectedSounds.length < 5) {
          const randomSound = possibleSounds[Math.floor(Math.random() * possibleSounds.length)];
          if (!detectedSounds.includes(randomSound)) {
            setDetectedSounds(prev => [...prev, randomSound]);
          }
        }
      }, 1500);

      // Progress through stages
      const progressStages = () => {
        if (currentStage < processingStages.length) {
          setProcessingStage(processingStages[currentStage].message);
          currentStage++;
          setTimeout(progressStages, processingStages[currentStage - 1]?.duration || 2000);
        }
      };

      progressStages();

      return () => {
        clearInterval(soundInterval);
      };
    } else {
      setDetectedSounds([]);
      setProcessingStage('');
    }
  }, [loading]);

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
      setNarration('');
      setAudioUrl('');
      setDetectedSounds([]);
    }
  };

  const handleProcess = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError('');
    setNarration('');
    setAudioUrl('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_URL}/process-audio`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setNarration(response.data.narration);
      setAudioUrl(`${API_URL}${response.data.audio_url}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred while processing the audio');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileName('');
    setNarration('');
    setAudioUrl('');
    setError('');
    setDetectedSounds([]);
    setProcessingStage('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            🎙️ Urban Sound Narrative Generator
          </h1>
          <p className="text-lg text-gray-600">
            Transform city sounds into vivid narratives
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
          
          {/* Upload Section */}
          {!loading && !narration && (
            <div className="p-8">
              <div className="border-3 border-dashed border-gray-300 rounded-xl p-12 text-center hover:border-indigo-400 transition-colors">
                <div className="mb-6">
                  <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                  </svg>
                </div>
                
                <label className="cursor-pointer">
                  <span className="mt-2 block text-sm font-medium text-gray-700">
                    {fileName || 'Choose an audio file to begin'}
                  </span>
                  <input
                    type="file"
                    accept=".mp3,.wav"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <span className="mt-4 inline-block bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors">
                    Browse Files
                  </span>
                </label>
                
                <p className="mt-4 text-xs text-gray-500">MP3 or WAV (Max 10MB)</p>
              </div>

              {error && (
                <div className="mt-6 bg-red-50 border-l-4 border-red-500 p-4 rounded">
                  <p className="text-red-700 font-medium">{error}</p>
                </div>
              )}

              {file && !error && (
                <div className="mt-8">
                  <button
                    onClick={handleProcess}
                    className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold py-4 px-6 rounded-xl transition-all transform hover:scale-105 shadow-lg"
                  >
                    🚀 Process Audio
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Processing Section */}
          {loading && (
            <div className="p-8 bg-gradient-to-br from-indigo-50 to-purple-50">
              <div className="text-center mb-8">
                <div className="inline-block relative">
                  <div className="animate-spin rounded-full h-20 w-20 border-4 border-indigo-200 border-t-indigo-600"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-2xl">🎵</span>
                  </div>
                </div>
                <h2 className="mt-6 text-2xl font-bold text-gray-800">
                  Processing Your Audio...
                </h2>
              </div>

              {/* Current Stage */}
              <div className="mb-8 text-center">
                <p className="text-lg font-semibold text-indigo-600 animate-pulse">
                  {processingStage}
                </p>
              </div>

              {/* Detected Sounds */}
              {detectedSounds.length > 0 && (
                <div className="bg-white rounded-xl p-6 shadow-lg">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    <span className="mr-2">🔊</span>
                    Detected Sounds
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {detectedSounds.map((sound, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-indigo-100 text-indigo-800 animate-fade-in"
                        style={{ animationDelay: `${index * 100}ms` }}
                      >
                        {sound}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Fun Facts */}
              <div className="mt-6 bg-purple-100 rounded-xl p-4 text-center">
                <p className="text-sm text-purple-800">
                  💡 <strong>Did you know?</strong> Our AI can identify over 500 different urban sounds!
                </p>
              </div>
            </div>
          )}

          {/* Results Section */}
          {narration && audioUrl && !loading && (
            <div className="p-8 space-y-6">
              <div className="text-center mb-6">
                <h2 className="text-3xl font-bold text-gray-900 mb-2">
                  ✨ Your Narrative is Ready!
                </h2>
                <p className="text-gray-600">Here's what we created from your audio</p>
              </div>

              {/* Narration Text */}
              <div className="bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-200 rounded-xl p-6 shadow-lg">
                <div className="flex items-start mb-3">
                  <span className="text-2xl mr-3">📝</span>
                  <h3 className="text-xl font-semibold text-gray-900">Generated Narration</h3>
                </div>
                <p className="text-gray-800 text-lg leading-relaxed italic">
                  "{narration}"
                </p>
              </div>

              {/* Audio Player */}
              <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-lg">
                <div className="flex items-start mb-4">
                  <span className="text-2xl mr-3">🎙️</span>
                  <h3 className="text-xl font-semibold text-gray-900">Audio Narration</h3>
                </div>
                
                <audio
                  controls
                  src={audioUrl}
                  className="w-full mb-4"
                  style={{ height: '54px' }}
                >
                  Your browser does not support the audio element.
                </audio>

                <div className="flex gap-3">
                  <a
                    href={audioUrl}
                    download="urban_narrative.mp3"
                    className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors text-center"
                  >
                    ⬇️ Download Audio
                  </a>
                  <button
                    onClick={handleReset}
                    className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                  >
                    🔄 Process Another
                  </button>
                </div>
              </div>

              {/* Share Section */}
              <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 text-center">
                <p className="text-sm text-yellow-800">
                  🎉 Love what you created? Share your urban narrative with the world!
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-600 text-sm">
          <p>Powered by AI • Whisper • Groq • ElevenLabs</p>
        </div>
      </div>

      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fade-in {
          animation: fade-in 0.5s ease-out forwards;
        }
      `}</style>
    </div>
  );
}

export default AudioProcessor;