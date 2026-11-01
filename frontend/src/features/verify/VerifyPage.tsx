import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  ModalityType,
  UnifiedVerificationResponse,
  SystemHealthResponse,
} from '../../types';
import {
  verifyImageFile,
  verifyVideoFile,
  verifyAudioFile,
  streamTextVerification,
  streamVideoVerification,
  PipelineProgressEvent,
} from '../../services/api';
import { LiveInvestigationPanel, LogEntry } from '../../components/LiveInvestigationPanel';
import { UnifiedReport } from '../../components/UnifiedReport';

interface VerifyPageProps {
  health: SystemHealthResponse | null;
  onAnalysisComplete?: (result: UnifiedVerificationResponse) => void;
}

export const VerifyPage: React.FC<VerifyPageProps> = ({ health, onAnalysisComplete }) => {
  // Mode selection: text | image | video | audio
  const [activeModality, setActiveModality] = useState<ModalityType>('text');

  // Input states
  const [claimText, setClaimText] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);

  // Analysis options
  const [verifyClaimsOption, setVerifyClaimsOption] = useState<boolean>(true);
  const [enableClassifierOption, setEnableClassifierOption] = useState<boolean>(true);
  const [whisperModelOption, setWhisperModelOption] = useState<string>('base');

  // Execution state
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [analysisStage, setAnalysisStage] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // SSE live investigation state (text-only)
  const [liveLogs, setLiveLogs] = useState<LogEntry[]>([]);
  const [liveStage, setLiveStage] = useState<string>('ingestion');
  const [liveStep, setLiveStep] = useState<number>(0);
  const [liveTotalSteps, setLiveTotalSteps] = useState<number>(8);
  const [showLivePanel, setShowLivePanel] = useState<boolean>(false);

  // Current result & history
  const [currentResult, setCurrentResult] = useState<UnifiedVerificationResponse | null>(null);
  const [sessionHistory, setSessionHistory] = useState<UnifiedVerificationResponse[]>([]);

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);
  // SSE cleanup ref
  const sseCleanupRef = useRef<(() => void) | null>(null);

  // Timer effect for analysis
  useEffect(() => {
    let timer: any;
    if (isAnalyzing) {
      setElapsedSeconds(0);
      timer = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isAnalyzing]);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => {
      if (sseCleanupRef.current) sseCleanupRef.current();
    };
  }, []);

  // Handle file selection
  const handleFileChange = (file: File | null) => {
    if (!file) {
      setSelectedFile(null);
      if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
      setFilePreviewUrl(null);
      return;
    }
    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setFilePreviewUrl(objectUrl);
  };

  // Switch modality and reset file
  const handleModalityChange = (modality: ModalityType) => {
    setActiveModality(modality);
    handleFileChange(null);
    setErrorMessage(null);
    setShowLivePanel(false);
  };

  // Preset claim triggers for fast testing
  const presetClaims = [
    'NASA confirmed the discovery of liquid water on Mars surface in September 2015.',
    'The Eiffel Tower was relocated to Lyon in 1944 during World War II.',
    'World Health Organization declared COVID-19 a global pandemic in March 2020.',
  ];

  // Append a new log entry (used by SSE callbacks)
  const appendLog = useCallback((evt: PipelineProgressEvent) => {
    setLiveLogs((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        stage: evt.stage,
        label: evt.label,
        icon: evt.icon,
        detail: evt.detail,
        step: evt.step,
        total_steps: evt.total_steps,
        ts: performance.now(),
      },
    ]);
    setLiveStage(evt.stage);
    setLiveStep(evt.step);
    setLiveTotalSteps(evt.total_steps);
  }, []);

  // Execute verification
  const handleRunVerification = async () => {
    setErrorMessage(null);
    setIsAnalyzing(true);
    setCurrentResult(null);

    // Reset live panel for text and video SSE
    if (activeModality === 'text' || activeModality === 'video') {
      setLiveLogs([]);
      setLiveStage('ingestion');
      setLiveStep(0);
      setShowLivePanel(true);
    }

    try {
      if (activeModality === 'text') {
        if (!claimText.trim()) {
          throw new Error('Please enter a claim to verify.');
        }

        // Cancel any previous stream
        if (sseCleanupRef.current) {
          sseCleanupRef.current();
          sseCleanupRef.current = null;
        }

        setAnalysisStage('Dispatching to real-time pipeline…');

        const cleanup = streamTextVerification(claimText.trim(), {
          onStatus: (evt) => {
            appendLog(evt);
          },
          onProgress: (evt) => {
            appendLog(evt);
            setAnalysisStage(evt.label);
          },
          onResult: (result) => {
            setCurrentResult(result);
            setSessionHistory((prev) => [result, ...prev]);
            if (onAnalysisComplete) onAnalysisComplete(result);
            setIsAnalyzing(false);
            setAnalysisStage('');
          },
          onError: (msg) => {
            setErrorMessage(msg);
            setIsAnalyzing(false);
            setAnalysisStage('');
          },
          onDone: () => {
            setIsAnalyzing(false);
            setAnalysisStage('');
          },
        });

        sseCleanupRef.current = cleanup;
        return; // SSE is async — don't fall through

      } else if (activeModality === 'image') {
        if (!selectedFile) throw new Error('Please select an image file to analyze.');
        setAnalysisStage('2. OCR & ViT AI-Image Detection');
        const result = await verifyImageFile(selectedFile);
        setCurrentResult(result);
        setSessionHistory((prev) => [result, ...prev]);
        if (onAnalysisComplete) onAnalysisComplete(result);

      } else if (activeModality === 'video') {
        if (!selectedFile) throw new Error('Please select a video file to analyze.');

        // Cancel any previous stream
        if (sseCleanupRef.current) {
          sseCleanupRef.current();
          sseCleanupRef.current = null;
        }

        setAnalysisStage('Uploading & dispatching video pipeline…');

        const cleanup = streamVideoVerification(
          selectedFile,
          {
            enableClassifier: enableClassifierOption,
            verifyClaims: verifyClaimsOption,
            whisperModel: whisperModelOption,
            maxOcrFrames: 15,
          },
          {
            onStatus: (evt) => {
              appendLog(evt);
              setAnalysisStage(evt.label);
            },
            onProgress: (evt) => {
              appendLog(evt);
              setAnalysisStage(evt.label);
            },
            onResult: (result) => {
              setCurrentResult(result);
              setSessionHistory((prev) => [result, ...prev]);
              if (onAnalysisComplete) onAnalysisComplete(result);
              setIsAnalyzing(false);
              setAnalysisStage('');
            },
            onError: (msg) => {
              setErrorMessage(msg);
              setIsAnalyzing(false);
              setAnalysisStage('');
            },
            onDone: () => {
              setIsAnalyzing(false);
              setAnalysisStage('');
            },
          }
        );

        sseCleanupRef.current = cleanup;
        return; // SSE is async — don't fall through

      } else {
        if (!selectedFile) throw new Error('Please select an audio file to analyze.');
        setAnalysisStage('2. Wav2Vec2 Synthetic Speech Detector & Whisper STT');
        const result = await verifyAudioFile(selectedFile, {
          verifyClaims: verifyClaimsOption,
          whisperModel: whisperModelOption,
        });
        setCurrentResult(result);
        setSessionHistory((prev) => [result, ...prev]);
        if (onAnalysisComplete) onAnalysisComplete(result);
      }
    } catch (err: any) {
      console.error('Verification error:', err);
      setErrorMessage(err.message || 'An unexpected error occurred during verification.');
    } finally {
      // For non-streaming modalities (image, audio), always clear analyzing state here
      if (activeModality !== 'text' && activeModality !== 'video') {
        setIsAnalyzing(false);
        setAnalysisStage('');
      }
    }
  };

  // Case identifier comes from the backend. Nothing is invented client-side.
  const currentTimestamp = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
  const caseId = currentResult?.case_summary?.case_id || 'PENDING';

  return (
    <div className="w-full flex flex-col items-center">
      <div className="w-full max-w-4xl px-4 sm:px-6 py-6 pb-24">
        {/* Desk Dispatch Subheader & Case Bar */}
        <div className="flex flex-col gap-1.5 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
              <span className="text-[11px] font-mono uppercase text-secondary tracking-widest font-semibold">
                Live Verification Session
              </span>
            </div>
            <span className="text-[10px] font-mono text-on-surface-variant bg-surface-container px-2 py-0.5 rounded border border-outline-variant/40">
              CASE: {caseId}
            </span>
          </div>

          <div className="flex items-baseline justify-between mt-1">
            <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl text-tertiary tracking-tight">
              Intake & Forensics
            </h1>
            <span className="text-[10px] font-mono text-outline uppercase tracking-wider">
              SEC-LEVEL 4
            </span>
          </div>
          <p className="text-xs sm:text-sm text-on-surface-variant leading-relaxed">
            Autonomous multimodal integrity parsing and evidence fusion against global news wires, sensor residues, and cryptographic manifests.
          </p>
        </div>

        {/* Multimodal Mode Selector (Segmented Control) */}
        <div className="w-full bg-surface-container-lowest p-1 rounded-xl mb-6 overflow-x-auto border border-outline-variant/40 shadow-sm">
          <div className="flex items-center min-w-max gap-1">
            <button
              type="button"
              onClick={() => handleModalityChange('text')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors flex items-center gap-1.5 ${
                activeModality === 'text'
                  ? 'bg-surface-container-high text-primary font-semibold shadow-sm border border-outline-variant/50'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">article</span>
              <span>Text Analysis</span>
            </button>

            <button
              type="button"
              onClick={() => handleModalityChange('image')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors flex items-center gap-1.5 ${
                activeModality === 'image'
                  ? 'bg-surface-container-high text-primary font-semibold shadow-sm border border-outline-variant/50'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">biotech</span>
              <span>Image Forensics</span>
            </button>

            <button
              type="button"
              onClick={() => handleModalityChange('video')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors flex items-center gap-1.5 ${
                activeModality === 'video'
                  ? 'bg-surface-container-high text-primary font-semibold shadow-sm border border-outline-variant/50'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">videocam</span>
              <span>Video Forensics</span>
            </button>

            <button
              type="button"
              onClick={() => handleModalityChange('audio')}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-colors flex items-center gap-1.5 ${
                activeModality === 'audio'
                  ? 'bg-surface-container-high text-primary font-semibold shadow-sm border border-outline-variant/50'
                  : 'text-on-surface-variant hover:text-on-surface'
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">graphic_eq</span>
              <span>Audio Deepfake</span>
            </button>
          </div>
        </div>

        {/* Source Ingestion Well */}
        <div className="bg-surface-container-low rounded-xl p-4 sm:p-5 mb-6 flex flex-col gap-3.5 border border-outline-variant/40 shadow-md">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-mono uppercase text-on-surface-variant tracking-wider font-semibold">
              Source Ingestion Well
            </span>
            <span className="text-[10px] font-mono text-primary flex items-center gap-1">
              <span className="material-symbols-outlined text-[14px]">sensors</span>
              <span>WELL ARMED // INTAKE READY</span>
            </span>
          </div>

          {/* Mode-Specific Input Panel */}
          {activeModality === 'text' && (
            <div className="flex flex-col gap-3">
              <div className="relative">
                <textarea
                  value={claimText}
                  onChange={(e) => setClaimText(e.target.value)}
                  placeholder="Enter a verifiable factual claim, news dispatch, or headline (e.g. 'NASA confirmed water on Mars')..."
                  rows={4}
                  className="w-full bg-surface-container-lowest border border-outline-variant/60 rounded-lg p-3 text-sm text-on-surface placeholder:text-outline focus:outline-none focus:border-primary transition-colors font-sans resize-y"
                />
                <div className="flex justify-between items-center mt-1 text-[10px] font-mono text-outline">
                  <span>Input Length: {claimText.length} characters</span>
                  <span>Minimum required: 5 chars</span>
                </div>
              </div>

              {/* Quick Sample Presets */}
              <div className="flex flex-col gap-1.5 pt-1">
                <span className="text-[10px] font-mono text-outline uppercase tracking-wider">
                  Preset Verification Claims:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {presetClaims.map((preset, idx) => (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setClaimText(preset)}
                      className="text-left text-[11px] font-mono px-2.5 py-1 rounded bg-surface-container-high/60 hover:bg-surface-container-highest text-on-surface-variant hover:text-on-surface transition-colors border border-outline-variant/30 truncate max-w-full"
                    >
                      "{preset.slice(0, 52)}..."
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeModality !== 'text' && (
            <div className="flex flex-col gap-3">
              {/* File Dropzone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-outline-variant/60 hover:border-primary/60 rounded-xl p-5 bg-surface-container-lowest/50 flex flex-col items-center justify-center gap-2 cursor-pointer transition-colors group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept={
                    activeModality === 'image'
                      ? 'image/jpeg,image/png,image/webp,image/bmp,image/tiff'
                      : activeModality === 'video'
                      ? 'video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm'
                      : 'audio/wav,audio/mpeg,audio/mp4,audio/flac,audio/ogg'
                  }
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileChange(e.target.files[0]);
                    }
                  }}
                />

                <div className="w-10 h-10 rounded-full bg-surface-container-high flex items-center justify-center text-primary group-hover:scale-105 transition-transform">
                  <span className="material-symbols-outlined text-[24px]">
                    {activeModality === 'image' ? 'add_photo_alternate' : activeModality === 'video' ? 'video_file' : 'audio_file'}
                  </span>
                </div>
                <div className="flex flex-col items-center text-center">
                  <span className="text-xs font-mono font-medium text-tertiary">
                    {selectedFile ? selectedFile.name : `Click or drag ${activeModality} file here`}
                  </span>
                  <span className="text-[10px] font-mono text-outline mt-0.5">
                    {selectedFile
                      ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for analysis`
                      : activeModality === 'image'
                      ? 'JPEG, PNG, WebP (Max 25MB)'
                      : activeModality === 'video'
                      ? 'MP4, MOV, MKV, WebM (Max 100MB)'
                      : 'WAV, MP3, FLAC, M4A, OGG (Max 50MB)'}
                  </span>
                </div>
              </div>

              {/* Media Preview if Selected */}
              {selectedFile && filePreviewUrl && (
                <div className="bg-surface-container rounded-lg p-3 flex flex-col sm:flex-row items-center justify-between gap-3 border border-outline-variant/40">
                  <div className="flex items-center gap-3 min-w-0 w-full sm:w-auto">
                    {activeModality === 'image' && (
                      <img
                        src={filePreviewUrl}
                        alt="Upload preview"
                        className="w-14 h-14 object-cover rounded-md border border-outline-variant flex-shrink-0"
                      />
                    )}
                    {activeModality === 'video' && (
                      <video
                        src={filePreviewUrl}
                        controls
                        className="w-28 h-16 object-cover rounded-md border border-outline-variant flex-shrink-0 bg-black"
                      />
                    )}
                    {activeModality === 'audio' && (
                      <audio src={filePreviewUrl} controls className="h-8 max-w-[200px] flex-shrink-0" />
                    )}
                    <div className="flex flex-col min-w-0 truncate">
                      <span className="text-xs font-mono text-tertiary font-medium truncate">
                        {selectedFile.name}
                      </span>
                      <span className="text-[10px] font-mono text-outline">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={() => handleFileChange(null)}
                    className="text-[11px] font-mono text-error hover:text-error/80 px-2 py-1 rounded bg-error-container/20 border border-error-container/30 transition-colors flex-shrink-0"
                  >
                    Remove File
                  </button>
                </div>
              )}

              {/* Execution Options for Video and Audio */}
              {(activeModality === 'video' || activeModality === 'audio') && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 text-xs font-mono bg-surface-container-lowest/60 p-2.5 rounded-lg border border-outline-variant/30">
                  <label className="flex items-center gap-2 cursor-pointer text-on-surface-variant hover:text-on-surface">
                    <input
                      type="checkbox"
                      checked={verifyClaimsOption}
                      onChange={(e) => setVerifyClaimsOption(e.target.checked)}
                      className="rounded accent-primary"
                    />
                    <span>Verify extracted spoken claims via web wire</span>
                  </label>

                  {activeModality === 'video' && (
                    <label className="flex items-center gap-2 cursor-pointer text-on-surface-variant hover:text-on-surface">
                      <input
                        type="checkbox"
                        checked={enableClassifierOption}
                        onChange={(e) => setEnableClassifierOption(e.target.checked)}
                        className="rounded accent-primary"
                      />
                      <span>Run VideoMAE visual classifier</span>
                    </label>
                  )}

                  <div className="flex items-center gap-2">
                    <span className="text-outline text-[11px]">Whisper STT:</span>
                    <select
                      value={whisperModelOption}
                      onChange={(e) => setWhisperModelOption(e.target.value)}
                      className="bg-surface-container-high border border-outline-variant/50 rounded px-1.5 py-0.5 text-[11px] text-tertiary"
                    >
                      <option value="tiny">tiny (fastest)</option>
                      <option value="base">base (balanced, default)</option>
                      <option value="small">small (accurate)</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Action Trigger Button */}
          <div className="flex items-center justify-between gap-3 pt-2">
            <div className="flex items-center gap-2 text-[10px] font-mono text-outline">
              <span className="material-symbols-outlined text-[14px]">shield</span>
              <span>Strict Signal Separation Enabled</span>
            </div>

            <button
              type="button"
              disabled={isAnalyzing}
              onClick={handleRunVerification}
              className={`h-11 px-5 rounded-lg font-mono text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 transition-all ${
                isAnalyzing
                  ? 'bg-surface-container-highest text-outline cursor-not-allowed'
                  : 'bg-primary-container hover:bg-primary text-on-primary shadow-sm hover:shadow active:scale-[0.99]'
              }`}
            >
              {isAnalyzing ? (
                <>
                  <span className="material-symbols-outlined text-[18px] animate-spin">progress_activity</span>
                  <span>Analyzing ({elapsedSeconds}s)...</span>
                </>
              ) : (
                <>
                  <span className="material-symbols-outlined text-[18px]">travel_explore</span>
                  <span>Execute Multimodal Verification</span>
                </>
              )}
            </button>
          </div>

          {/* Non-text processing banner */}
          {isAnalyzing && activeModality !== 'text' && (
            <div className="mt-2 p-3 rounded-lg bg-surface-container-high/80 border border-primary/30 flex flex-col gap-1.5 animate-pulse">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-primary font-semibold flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-[16px]">sync</span>
                  {analysisStage || 'Orchestrating forensic analysis...'}
                </span>
                <span className="text-outline">{elapsedSeconds}s elapsed</span>
              </div>
              <p className="text-[11px] text-on-surface-variant font-sans">
                Processing pipeline without freezing. Models execute on local host (GPU RTX 3050 / PyTorch) and live wire queries dispatch via Tavily.
              </p>
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && (
            <div className="mt-2 p-3 rounded-lg bg-error-container/30 border border-error/40 text-on-error-container flex items-start gap-2.5">
              <span className="material-symbols-outlined text-error text-[18px] flex-shrink-0 mt-0.5">
                error
              </span>
              <div className="flex flex-col gap-0.5 text-xs font-sans">
                <span className="font-semibold font-mono uppercase text-error">Ingestion or Verification Fault</span>
                <span>{errorMessage}</span>
              </div>
            </div>
          )}
        </div>

        {/* ──────────────────────────────────────────────────────────────── */}
        {/* LIVE INVESTIGATION PANEL (text + video modality, SSE active) */}
        {/* ──────────────────────────────────────────────────────────────── */}
        {showLivePanel && (activeModality === 'text' || activeModality === 'video') && (
          <div className="bg-surface-container-low rounded-xl p-4 sm:p-5 mb-6 border border-outline-variant/40 shadow-md">
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono uppercase tracking-wider text-tertiary font-semibold flex items-center gap-1.5">
                <span className="material-symbols-outlined text-[16px] text-secondary">sensors</span>
                Investigation Pipeline
              </span>
              {!isAnalyzing && (
                <button
                  type="button"
                  onClick={() => setShowLivePanel(false)}
                  className="text-[10px] font-mono text-outline hover:text-on-surface transition-colors flex items-center gap-1"
                >
                  <span className="material-symbols-outlined text-[14px]">close</span>
                  Hide
                </button>
              )}
            </div>
            <LiveInvestigationPanel
              logs={liveLogs}
              currentStage={liveStage}
              currentStep={liveStep}
              totalSteps={liveTotalSteps}
              elapsedSeconds={elapsedSeconds}
              isRunning={isAnalyzing}
              hasError={!!errorMessage}
            />
          </div>
        )}

                {/* RESULTS PANEL — one clear, evidence-based final report */}
        {currentResult && (
          <div className="flex flex-col gap-6 mt-6">
            {/* Unified final report: verdict, confidence, why, evidence, limitations. */}
            <UnifiedReport
              result={currentResult}
              modality={activeModality}
              claimText={claimText}
              selectedFile={selectedFile}
            />
          </div>
        )}
      </div>
    </div>
  );
};
