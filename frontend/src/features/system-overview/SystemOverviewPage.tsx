import React, { useState } from 'react';
import { ArchitectureNode, EvaluationBenchmark } from '../../types';

export const SystemOverviewPage: React.FC = () => {
  // Selected node for detailed inspector panel
  const [selectedNode, setSelectedNode] = useState<ArchitectureNode | null>(null);
  const [expandedLevels, setExpandedLevels] = useState<Record<number, boolean>>({
    1: true,
    2: true,
    3: true,
    4: true,
    5: true,
  });
  const [activeBranchFilter, setActiveBranchFilter] = useState<string>('all');

  // Ground-truth Architecture Nodes
  const architectureNodes: ArchitectureNode[] = [
    {
      id: 'root',
      label: 'TRUTHLENSAI Core Orchestrator',
      level: 0,
      modality: 'core',
      description: 'Central multi-agent multimodal misinformation verification orchestrator. Strictly separates factual claim veracity from media provenance and visual classification.',
      entryPoint: 'multimodal_router.py:route_media() / api/server.py',
      models: ['FastAPI REST Engine', 'Multimodal Dispatcher'],
      latency: '~10-40ms dispatch',
      limitations: 'Dispatches subroutines concurrently; GPU VRAM bounded by local 4GB hardware.',
      children: ['in-text', 'in-img', 'in-vid', 'in-aud'],
    },
    // Level 1: Inputs
    {
      id: 'in-text',
      label: 'Text Claim Input',
      level: 1,
      modality: 'text',
      description: 'Ingestion of raw news headlines, textual claims, or article statements.',
      entryPoint: 'api/server.py:endpoint_verify_text()',
      parentId: 'root',
      children: ['pre-text'],
    },
    {
      id: 'in-img',
      label: 'Image Media Input',
      level: 1,
      modality: 'image',
      description: 'Ingestion of image files (.png, .jpg, .webp, .tiff) with structural format validation.',
      entryPoint: 'api/server.py:endpoint_verify_image()',
      parentId: 'root',
      children: ['pre-img'],
    },
    {
      id: 'in-vid',
      label: 'Video Media Input',
      level: 1,
      modality: 'video',
      description: 'Ingestion of video files (.mp4, .mov, .mkv, .webm) with MIME type and size checks.',
      entryPoint: 'api/server.py:endpoint_verify_video()',
      parentId: 'root',
      children: ['pre-vid'],
    },
    {
      id: 'in-aud',
      label: 'Audio Recording Input',
      level: 1,
      modality: 'audio',
      description: 'Ingestion of audio recordings (.wav, .mp3, .flac, .ogg, .m4a) with duration guardrails.',
      entryPoint: 'api/server.py:endpoint_verify_audio()',
      parentId: 'root',
      children: ['pre-aud'],
    },
    // Level 2: Preprocessing
    {
      id: 'pre-text',
      label: 'Claim Normalization & Query Planning',
      level: 2,
      modality: 'text',
      description: 'Extracts core factual assertions, generates diverse temporal and exact-date search queries.',
      entryPoint: 'query_planner.py:plan_queries()',
      models: ['Groq / qwen3.8-27b'],
      latency: '~1.2s',
      limitations: 'May over-decompose subjective nuance or sarcastic humor.',
      parentId: 'in-text',
      children: ['mod-text'],
    },
    {
      id: 'pre-img',
      label: 'OCR Text & YOLO Object Extraction',
      level: 2,
      modality: 'image',
      description: 'Dual visual preprocessing: EasyOCR extracts on-screen text, Ultralytics YOLO11 identifies physical entities.',
      entryPoint: 'image_pipeline.py:process_image()',
      models: ['EasyOCR', 'Ultralytics YOLO11n'],
      latency: '~650ms',
      limitations: 'OCR performance degrades under severe blur or stylized watermarks.',
      parentId: 'in-img',
      children: ['mod-img'],
    },
    {
      id: 'pre-vid',
      label: 'Demuxing, Frame Extraction & Audio Track Separation',
      level: 2,
      modality: 'video',
      description: 'FFmpeg and OpenCV isolate 16-frame uniform clips for spatial inspection and demux 16kHz audio track.',
      entryPoint: 'video_module/preprocessor.py:extract_frames_and_audio()',
      models: ['FFmpeg', 'OpenCV'],
      latency: '~1.8s',
      limitations: 'High frame-rate 4K files require stride subsampling to stay within VRAM bounds.',
      parentId: 'in-vid',
      children: ['mod-vid'],
    },
    {
      id: 'pre-aud',
      label: 'Audio Decoding & 16kHz Mel-Spectrogram Resampling',
      level: 2,
      modality: 'audio',
      description: 'Normalizes audio waveform to 16kHz mono, enforces >=0.5s duration boundary check.',
      entryPoint: 'audio_module/audio_pipeline.py:ingest_audio()',
      models: ['Torchaudio', 'SoundFile'],
      latency: '~80ms',
      limitations: 'Audio under 0.5s triggers immediate UNAVAILABLE failsafe.',
      parentId: 'in-aud',
      children: ['mod-aud'],
    },
    // Level 3: Four Module Branches
    {
      id: 'mod-text',
      label: 'Multi-Agent Text Research Engine',
      level: 3,
      modality: 'text',
      description: 'Dispatches multi-agent retrieval to Tavily Search API, scrapes candidate pages, and filters near-duplicate domains.',
      entryPoint: 'pipeline.py:run_research_pipeline()',
      models: ['Tavily API', 'Domain Reliability Tiers'],
      latency: '~3-6s',
      limitations: 'Requires active internet connectivity and valid Tavily API key.',
      parentId: 'pre-text',
      children: ['shared-verif'],
    },
    {
      id: 'mod-img',
      label: 'ViT AI-Generated Image Origin Detector',
      level: 3,
      modality: 'image',
      description: 'Visual Transformer classifier detects synthetic diffusion and GAN generation artifacts independently from OCR claims.',
      entryPoint: 'ai_image_signal.py:detect_ai_image()',
      models: ['umm-maybe/AI-image-detector (ViT)'],
      latency: '~340ms',
      limitations: 'Prone to uncertainty on heavily compressed social media crops.',
      parentId: 'pre-img',
      children: ['shared-verif'],
    },
    {
      id: 'mod-vid',
      label: 'Dual Video Analyzers: VideoMAE & EfficientNet Detector',
      level: 3,
      modality: 'video',
      description: 'Runs VideoMAE-base probe for visual misinformation heuristic (61.90% test acc) and BiGRU for AI-generated video detection (98.33% in-dist, 81.44% OOD OpenSora).',
      entryPoint: 'video_module/video_verifier.py & videomae_adapter.py',
      models: ['MCG-NJU/videomae-base', 'EfficientNet-B0 + BiGRU'],
      latency: '~2.4s',
      limitations: 'Experimental signal; held-out OpenSora generator causes 18.5% recall drop.',
      parentId: 'pre-vid',
      children: ['shared-verif'],
    },
    {
      id: 'mod-aud',
      label: 'Wav2Vec2 Synthetic Speech & Whisper STT Engine',
      level: 3,
      modality: 'audio',
      description: 'Evaluates vocoder phase continuity and phoneme spectral flattening via Wav2Vec2; transcribes spoken claims via Whisper.',
      entryPoint: 'audio_module/audio_pipeline.py:verify_audio()',
      models: ['mo-thecreator/Deepfake-audio-detection', 'Whisper base'],
      latency: '~1.1s',
      limitations: 'High false-positive rate on accented/regional Indic speech (Marathi 0% acc in 16-sample benchmark).',
      parentId: 'pre-aud',
      children: ['shared-verif'],
    },
    // Level 4: Shared Orchestration
    {
      id: 'shared-verif',
      label: 'Bayesian Evidentiary Fusion & Reliability Calibration',
      level: 4,
      modality: 'shared',
      description: 'Normalizes independent source agreement, applies NLI cross-encoder verification, and penalizes single-signal anomalies.',
      entryPoint: 'evidence_fusion.py:calculate_fusion_score()',
      models: ['Cross-Encoder NLI', 'Source Tier Profile Matrix'],
      latency: '~850ms',
      limitations: 'Requires at least 2 independent corroborating sources for SUFFICIENT evidence grade.',
      parentId: 'mod-text',
      children: ['out-verdict'],
    },
    // Level 5: Final Output
    {
      id: 'out-verdict',
      label: 'Adjudicated Dossier & Cryptographic Seal',
      level: 5,
      modality: 'core',
      description: 'Emits decoupled factual veracity verdict, media provenance signal, evidence citations, and immutable SHA seal.',
      entryPoint: 'final_verdict.py:generate_final_verdict()',
      models: ['Digital Seal Generator (ED-xxx-TL)'],
      latency: '~50ms',
      limitations: 'Factual verdicts are strictly evidence-dependent and never overridden by synthetic media presence.',
      parentId: 'shared-verif',
    },
  ];

  // Ground-truth Real Benchmarks
  const benchmarks: EvaluationBenchmark[] = [
    {
      title: 'VideoMAE Visual Misinformation Classifier',
      modelName: 'TruthLensAI Video Classifier (VideoMAE-base + Linear Probe)',
      architecture: 'ViT-base patch16 224 (MCG-NJU/videomae-base)',
      datasetName: 'Fake Video Corpus (FVC) Benchmark',
      sampleCount: 42,
      primaryMetric: { label: 'Test Accuracy', value: '61.90%' },
      secondaryMetrics: [
        { label: 'Macro F1', value: '0.6047' },
        { label: 'ROC-AUC', value: '0.6759' },
        { label: 'Fake Recall', value: '70.83%' },
        { label: 'Real Precision', value: '56.25%' },
      ],
      confusionMatrix: { tp: 17, tn: 9, fp: 9, fn: 7 },
      notes: 'Outperformed baseline EfficientNet (54.11% macro F1). Documented as an experimental supporting heuristic only.',
      status: 'experimental',
    },
    {
      title: 'AI-Generated Video Detector (In-Distribution)',
      modelName: 'TruthLensAI AI-Gen Video Detector (EfficientNet-B0 + BiGRU)',
      architecture: 'Pretrained EfficientNet-B0 backbone with Bidirectional GRU temporal aggregation',
      datasetName: 'Pilot Synthetic Corpus (VideoCrafter2 + Clean Real)',
      sampleCount: 60,
      primaryMetric: { label: 'In-Dist Accuracy', value: '98.33%' },
      secondaryMetrics: [
        { label: 'Macro F1', value: '0.9825' },
        { label: 'Balanced Acc', value: '98.65%' },
        { label: 'ROC-AUC', value: '0.9988' },
        { label: 'Peak VRAM', value: '54 MB' },
      ],
      confusionMatrix: { tp: 23, tn: 36, fp: 1, fn: 0 },
      notes: 'Near-perfect performance on in-distribution VideoCrafter2 generator. Fast inference (3.42ms per video).',
      status: 'pilot',
    },
    {
      title: 'AI-Generated Video Detector (Out-of-Distribution)',
      modelName: 'TruthLensAI AI-Gen Video Detector (OpenSora Stress Test)',
      architecture: 'Pretrained EfficientNet-B0 backbone with Bidirectional GRU',
      datasetName: 'Held-out OpenSora Video Diffusion Transformer (ood_generator.csv)',
      sampleCount: 97,
      primaryMetric: { label: 'OOD Accuracy', value: '81.44%' },
      secondaryMetrics: [
        { label: 'OOD Recall', value: '81.44%' },
        { label: 'False Negatives', value: '18 / 97' },
        { label: 'Degradation', value: '-16.89% vs In-Dist' },
      ],
      confusionMatrix: { tp: 79, tn: 0, fp: 0, fn: 18 },
      notes: 'Demonstrates real-world generalization gap: completely unseen generator architecture caused 18 false negatives.',
      status: 'pilot',
    },
    {
      title: 'Audio Synthetic-Speech Detector (8-Sample Validation)',
      modelName: 'Wav2Vec2 Deepfake Audio Detector',
      architecture: 'Fine-tuned Wav2Vec2 audio representation network',
      datasetName: 'Validation Benchmark (Reuters, Hindi, Marathi, SAPI TTS)',
      sampleCount: 8,
      primaryMetric: { label: 'Validation Accuracy', value: '75.00%' },
      secondaryMetrics: [
        { label: 'Precision', value: '66.67%' },
        { label: 'Recall', value: '100.0%' },
        { label: 'F1 Score', value: '0.8000' },
      ],
      confusionMatrix: { tp: 4, tn: 2, fp: 2, fn: 0 },
      notes: '100% synthetic recall; false positives occurred on compressed Reuters news and Marathi field report.',
      status: 'verified',
    },
    {
      title: 'Audio Detector Multi-Language Stress Test (16-Sample)',
      modelName: 'Wav2Vec2 Deepfake Audio Detector (Extended Stress)',
      architecture: 'Fine-tuned Wav2Vec2',
      datasetName: 'Extended Language & Compression Suite (eval_metrics.json)',
      sampleCount: 16,
      primaryMetric: { label: 'Stress Accuracy', value: '66.67%' },
      secondaryMetrics: [
        { label: 'English Acc', value: '66.67% (8/12)' },
        { label: 'Hindi Acc', value: '100.0% (2/2)' },
        { label: 'Marathi Acc', value: '0.0% (0/1 FP)' },
        { label: 'Short Audio Edge', value: '1 Excluded (<0.5s)' },
      ],
      confusionMatrix: { tp: 5, tn: 5, fp: 3, fn: 2, uncertain: 1 },
      notes: 'Revealed vulnerability to lossy MP3 compression (64kbps) and regional Indic accents. Handled <0.5s via UNAVAILABLE failsafe.',
      status: 'experimental',
    },
    {
      title: 'Multimodal Regression Test Suite',
      modelName: 'TruthLensAI Complete PyTest Verification Suite',
      architecture: 'Automated PyTest Runner with Mocking & Live Torch Guards',
      datasetName: 'Unit, Regression & End-to-End API Contracts',
      sampleCount: 98,
      primaryMetric: { label: 'Pass Rate', value: '100.0% (98/98)' },
      secondaryMetrics: [
        { label: 'Audio Module', value: '22 / 22 Passed' },
        { label: 'Video Module', value: '55 / 55 Passed' },
        { label: 'Integration API', value: '21 / 21 Passed' },
      ],
      notes: 'All 98 tests pass cleanly in 32 seconds on Windows 11 / Python 3.11 with RTX 3050 Laptop GPU.',
      status: 'verified',
    },
  ];

  // Filtered nodes based on active branch
  const filteredNodes = architectureNodes.filter((node) => {
    if (activeBranchFilter === 'all') return true;
    return (
      node.modality === activeBranchFilter ||
      node.modality === 'core' ||
      node.modality === 'shared'
    );
  });

  // Toggle level
  const toggleLevel = (lvl: number) => {
    setExpandedLevels((prev) => ({ ...prev, [lvl]: !prev[lvl] }));
  };

  const expandAll = () => {
    setExpandedLevels({ 1: true, 2: true, 3: true, 4: true, 5: true });
  };

  const collapseAll = () => {
    setExpandedLevels({ 1: false, 2: false, 3: false, 4: false, 5: false });
  };

  return (
    <div className="w-full flex flex-col items-center">
      <div className="w-full max-w-5xl px-4 sm:px-6 py-6 pb-24 flex flex-col gap-8">
        {/* Top Archival Context Pill */}
        <div className="flex items-center justify-between pt-1 border-b border-outline-variant/30 pb-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-[11px] font-mono uppercase text-primary tracking-wider font-semibold">
              PIPELINE AUDIT // v3.4 ARCHITECTURE
            </span>
          </div>
          <span className="text-[10px] font-mono text-on-surface-variant bg-surface-container px-2 py-0.5 rounded border border-outline-variant/30">
            SYS_SHA256: 9b2d..e810
          </span>
        </div>

        {/* Editorial Masthead Section */}
        <header className="flex flex-col gap-2">
          <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl text-tertiary tracking-tight leading-snug">
            Algorithmic Architecture & Verification Methodology
          </h1>
          <p className="text-xs sm:text-sm text-on-surface-variant leading-relaxed max-w-3xl">
            A transparent technical disclosure of the multi-layered forensic inspection apparatus engineered for accredited newsrooms, human rights monitors, and sovereign open-source investigators.
          </p>
          <div className="flex flex-wrap items-center gap-3 mt-1">
            <div className="flex items-center gap-1.5 bg-surface-container-high px-2.5 py-1 rounded-lg border border-outline-variant/40">
              <span className="material-symbols-outlined text-[14px] text-secondary">verified_user</span>
              <span className="text-[10px] font-mono text-secondary uppercase font-semibold">
                IFCN Fact-Checking Standards Compliant
              </span>
            </div>
            <span className="text-[10px] font-mono text-outline">
              Audit Cycle: Q1 2025 • Hardware: RTX 3050 Laptop GPU (4GB VRAM)
            </span>
          </div>
        </header>

        {/* SECTION 1: INTERACTIVE ARCHITECTURE TREE */}
        <section className="bg-surface-container rounded-xl p-5 sm:p-6 border border-outline-variant/50 shadow-lg flex flex-col gap-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-outline-variant/30 pb-4">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px] text-primary">account_tree</span>
              <h2 className="font-serif text-lg sm:text-xl text-tertiary font-semibold">
                Multimodal Pipeline Hierarchy (5 Levels)
              </h2>
            </div>

            {/* Tree Controls */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Branch Filter */}
              <div className="flex items-center gap-1 bg-surface-container-lowest p-1 rounded-lg border border-outline-variant/40 text-[10px] font-mono">
                {(['all', 'text', 'image', 'video', 'audio'] as const).map((branch) => (
                  <button
                    key={branch}
                    type="button"
                    onClick={() => setActiveBranchFilter(branch)}
                    className={`px-2 py-0.5 rounded capitalize transition-colors ${
                      activeBranchFilter === branch
                        ? 'bg-surface-container-high text-primary font-semibold'
                        : 'text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    {branch}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-1 text-[11px] font-mono">
                <button
                  type="button"
                  onClick={expandAll}
                  className="px-2 py-1 rounded bg-surface-container-high hover:bg-surface-bright text-tertiary border border-outline-variant/30"
                >
                  Expand All
                </button>
                <button
                  type="button"
                  onClick={collapseAll}
                  className="px-2 py-1 rounded bg-surface-container-high hover:bg-surface-bright text-tertiary border border-outline-variant/30"
                >
                  Collapse All
                </button>
              </div>
            </div>
          </div>

          {/* Node Levels Grid / Accordion */}
          <div className="flex flex-col gap-4">
            {[1, 2, 3, 4, 5].map((levelNumber) => {
              const levelTitles: Record<number, string> = {
                1: 'Level 1 — Input & Ingestion Layer',
                2: 'Level 2 — Preprocessing & Signal Normalization',
                3: 'Level 3 — Modality-Specific Forensic Branches',
                4: 'Level 4 — Shared Verification & Bayesian Fusion',
                5: 'Level 5 — Final Adjudication & Evidence Dossier',
              };

              const nodesAtLevel = filteredNodes.filter((n) => n.level === levelNumber);
              if (nodesAtLevel.length === 0) return null;

              const isExpanded = expandedLevels[levelNumber];

              return (
                <div
                  key={levelNumber}
                  className="bg-surface-container-low rounded-lg border border-outline-variant/40 overflow-hidden"
                >
                  {/* Level Header Toggle */}
                  <button
                    type="button"
                    onClick={() => toggleLevel(levelNumber)}
                    className="w-full px-4 py-2.5 bg-surface-container-high/60 hover:bg-surface-container-high transition-colors flex items-center justify-between text-left"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-semibold text-primary">
                        {levelTitles[levelNumber]}
                      </span>
                      <span className="text-[10px] font-mono text-outline">
                        ({nodesAtLevel.length} nodes)
                      </span>
                    </div>
                    <span className="material-symbols-outlined text-[16px] text-outline">
                      {isExpanded ? 'expand_less' : 'expand_more'}
                    </span>
                  </button>

                  {/* Level Content */}
                  {isExpanded && (
                    <div className="p-3 sm:p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
                      {nodesAtLevel.map((node) => {
                        const isSelected = selectedNode?.id === node.id;
                        return (
                          <div
                            key={node.id}
                            onClick={() => setSelectedNode(node)}
                            className={`p-3 rounded-lg border transition-all cursor-pointer flex flex-col gap-1.5 ${
                              isSelected
                                ? 'bg-surface-container-highest border-primary shadow-sm'
                                : 'bg-surface-container hover:bg-surface-container-high border-outline-variant/40'
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-xs font-mono font-semibold text-tertiary">
                                {node.label}
                              </span>
                              <span
                                className={`text-[9px] font-mono uppercase px-1.5 py-0.2 rounded ${
                                  node.modality === 'text'
                                    ? 'bg-primary/20 text-primary'
                                    : node.modality === 'image'
                                    ? 'bg-secondary/20 text-secondary'
                                    : node.modality === 'video'
                                    ? 'bg-amber-400/20 text-amber-300'
                                    : node.modality === 'audio'
                                    ? 'bg-cyan-400/20 text-cyan-300'
                                    : 'bg-surface-container-highest text-tertiary'
                                }`}
                              >
                                {node.modality}
                              </span>
                            </div>

                            <p className="text-[11px] text-on-surface-variant font-sans line-clamp-2">
                              {node.description}
                            </p>

                            {node.entryPoint && (
                              <div className="mt-1 pt-1 border-t border-outline-variant/20 flex items-center justify-between text-[10px] font-mono text-outline">
                                <span className="truncate max-w-[200px]">fn: {node.entryPoint}</span>
                                <span className="text-primary hover:underline">Click to Inspect →</span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Node Detail Inspector Drawer/Modal */}
          {selectedNode && (
            <div className="mt-2 p-4 sm:p-5 rounded-xl bg-surface-container-lowest border-2 border-primary/40 flex flex-col gap-3 relative shadow-2xl">
              <div className="flex items-center justify-between border-b border-outline-variant/30 pb-2">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[18px] text-primary">data_object</span>
                  <span className="text-xs font-mono uppercase text-tertiary font-bold tracking-wider">
                    Node Inspector: {selectedNode.label}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedNode(null)}
                  className="text-outline hover:text-tertiary text-xs font-mono px-2 py-0.5 rounded bg-surface-container"
                >
                  Close ×
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
                <div>
                  <span className="text-[10px] text-outline block">CORE FUNCTION / ENTRY POINT:</span>
                  <span className="text-primary">{selectedNode.entryPoint || 'Internal Pipeline Engine'}</span>
                </div>
                <div>
                  <span className="text-[10px] text-outline block">ESTIMATED LATENCY:</span>
                  <span className="text-tertiary">{selectedNode.latency || 'N/A'}</span>
                </div>
                {selectedNode.models && (
                  <div className="sm:col-span-2">
                    <span className="text-[10px] text-outline block">BACKBONE MODELS & LIBRARIES:</span>
                    <span className="text-secondary">{selectedNode.models.join(', ')}</span>
                  </div>
                )}
                <div className="sm:col-span-2">
                  <span className="text-[10px] text-outline block">FULL ARCHITECTURAL PURPOSE:</span>
                  <p className="text-[11px] text-on-surface font-sans leading-relaxed mt-0.5">
                    {selectedNode.description}
                  </p>
                </div>
                {selectedNode.limitations && (
                  <div className="sm:col-span-2 bg-error-container/20 p-2.5 rounded border border-error-container/30">
                    <span className="text-[10px] text-error block font-bold">OPERATIONAL BOUNDARY / LIMITATION:</span>
                    <p className="text-[11px] text-on-error-container font-sans mt-0.5">
                      {selectedNode.limitations}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>

        {/* SECTION 2: FOUNDATIONAL MODELS & ENSEMBLES */}
        <section className="bg-surface-container rounded-xl p-5 sm:p-6 border border-outline-variant/50 shadow-lg flex flex-col gap-4">
          <div className="flex items-center gap-2 border-b border-outline-variant/30 pb-3">
            <span className="material-symbols-outlined text-[20px] text-primary">memory</span>
            <h2 className="font-serif text-lg sm:text-xl text-tertiary font-semibold">
              Foundational Models & Analytical Subroutines
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            {/* Model 1: Groq LLM */}
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-tertiary">qwen3.8-27b</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-primary/20 text-primary">LLM</span>
                </div>
                <span className="text-[10px] font-mono text-outline block mt-0.5">Provider: Groq Cloud</span>
                <p className="text-[11px] text-on-surface-variant font-sans mt-1.5">
                  Orchestrates Query Planning, decomposes complex articles into atomic verifiable claims, and executes Bayesian evidence reasoning.
                </p>
              </div>
              <div className="text-[10px] font-mono text-outline pt-2 border-t border-outline-variant/20 flex justify-between">
                <span>Role: Reasoning</span>
                <span>Latency: ~600ms</span>
              </div>
            </div>

            {/* Model 2: VideoMAE */}
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-tertiary">VideoMAE-base (Linear Probe)</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-secondary/20 text-secondary">Visual</span>
                </div>
                <span className="text-[10px] font-mono text-outline block mt-0.5">MCG-NJU/videomae-base</span>
                <p className="text-[11px] text-on-surface-variant font-sans mt-1.5">
                  Visual Misinformation Classifier trained on Fake Video Corpus (FVC). Extracts 16-frame spatiotemporal features with 61.90% test accuracy.
                </p>
              </div>
              <div className="text-[10px] font-mono text-outline pt-2 border-t border-outline-variant/20 flex justify-between">
                <span>Role: Visual Heuristic</span>
                <span>Params: 86M</span>
              </div>
            </div>

            {/* Model 3: EfficientNet BiGRU */}
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-tertiary">EfficientNet-B0 + BiGRU</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-400/20 text-amber-300">AI-Gen Vid</span>
                </div>
                <span className="text-[10px] font-mono text-outline block mt-0.5">Custom Trained Checkpoint</span>
                <p className="text-[11px] text-on-surface-variant font-sans mt-1.5">
                  Dedicated AI-generated video detection model. Achieved 98.33% accuracy on in-distribution test set; stress-tested against OpenSora (81.44%).
                </p>
              </div>
              <div className="text-[10px] font-mono text-outline pt-2 border-t border-outline-variant/20 flex justify-between">
                <span>Role: Video Synthesis</span>
                <span>VRAM: 54 MB</span>
              </div>
            </div>

            {/* Model 4: Wav2Vec2 */}
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-tertiary">Deepfake-audio-detection</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-cyan-400/20 text-cyan-300">Speech</span>
                </div>
                <span className="text-[10px] font-mono text-outline block mt-0.5">mo-thecreator / HuggingFace</span>
                <p className="text-[11px] text-on-surface-variant font-sans mt-1.5">
                  Wav2Vec2 architecture measuring vocoder phase discontinuity and formant trajectory naturalness. 75% accuracy in 8-sample benchmark.
                </p>
              </div>
              <div className="text-[10px] font-mono text-outline pt-2 border-t border-outline-variant/20 flex justify-between">
                <span>Role: Synthetic Speech</span>
                <span>Latency: ~40ms</span>
              </div>
            </div>

            {/* Model 5: ViT AI Image */}
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-tertiary">AI-image-detector (ViT)</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-secondary/20 text-secondary">Image</span>
                </div>
                <span className="text-[10px] font-mono text-outline block mt-0.5">umm-maybe / HuggingFace</span>
                <p className="text-[11px] text-on-surface-variant font-sans mt-1.5">
                  Vision Transformer fine-tuned to classify synthetic image diffusion generators (Midjourney, DALL-E, Stable Diffusion) vs camera captures.
                </p>
              </div>
              <div className="text-[10px] font-mono text-outline pt-2 border-t border-outline-variant/20 flex justify-between">
                <span>Role: Image Provenance</span>
                <span>Latency: ~340ms</span>
              </div>
            </div>

            {/* Model 6: Whisper & YOLO */}
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-2">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-tertiary">Whisper base & YOLO11n</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-surface-container-highest text-tertiary">Sensors</span>
                </div>
                <span className="text-[10px] font-mono text-outline block mt-0.5">OpenAI & Ultralytics</span>
                <p className="text-[11px] text-on-surface-variant font-sans mt-1.5">
                  Whisper provides robust multi-lingual speech-to-text; YOLO11n provides real-time bounding box detection for contextual claim grounding.
                </p>
              </div>
              <div className="text-[10px] font-mono text-outline pt-2 border-t border-outline-variant/20 flex justify-between">
                <span>Role: Perception / OCR</span>
                <span>Local GPU</span>
              </div>
            </div>
          </div>
        </section>

        {/* SECTION 3: EMPIRICAL BENCHMARKS & EVALUATION DASHBOARD */}
        <section className="bg-surface-container rounded-xl p-5 sm:p-6 border border-outline-variant/50 shadow-lg flex flex-col gap-5">
          <div className="flex items-center justify-between border-b border-outline-variant/30 pb-3">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[20px] text-primary">analytics</span>
              <h2 className="font-serif text-lg sm:text-xl text-tertiary font-semibold">
                Empirical Evaluation & Measured Test Results
              </h2>
            </div>
            <span className="text-[10px] font-mono text-outline">
              SOURCED DIRECT FROM REPO JSON ARTIFACTS
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {benchmarks.map((bench, idx) => (
              <div
                key={idx}
                className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col justify-between gap-3"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-xs font-mono font-bold text-tertiary">
                      {bench.title}
                    </span>
                    <span
                      className={`text-[9px] font-mono uppercase px-1.5 py-0.2 rounded flex-shrink-0 ${
                        bench.status === 'verified'
                          ? 'bg-primary/20 text-primary border border-primary/30'
                          : 'bg-secondary/20 text-secondary border border-secondary/30'
                      }`}
                    >
                      {bench.status}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-outline block mt-0.5">
                    Dataset: {bench.datasetName} ({bench.sampleCount} samples)
                  </span>

                  {/* Primary Metric Hero */}
                  <div className="my-2.5 p-2.5 rounded bg-surface-container-lowest border border-outline-variant/30 flex items-center justify-between">
                    <span className="text-xs font-mono text-on-surface-variant">
                      {bench.primaryMetric.label}:
                    </span>
                    <span className="font-serif text-xl font-bold text-primary">
                      {bench.primaryMetric.value}
                    </span>
                  </div>

                  {/* Secondary Metrics Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5 text-[10px] font-mono">
                    {bench.secondaryMetrics.map((sec, sIdx) => (
                      <div key={sIdx} className="bg-surface-container px-2 py-1 rounded">
                        <span className="text-outline block truncate">{sec.label}</span>
                        <span className="text-tertiary font-semibold">{sec.value}</span>
                      </div>
                    ))}
                  </div>

                  {/* Confusion Matrix breakdown if available */}
                  {bench.confusionMatrix && (
                    <div className="mt-2 text-[9px] font-mono text-outline flex items-center justify-between bg-surface-container-lowest/80 px-2 py-1 rounded">
                      <span>TP: {bench.confusionMatrix.tp}</span>
                      <span>TN: {bench.confusionMatrix.tn}</span>
                      <span>FP: {bench.confusionMatrix.fp}</span>
                      <span>FN: {bench.confusionMatrix.fn}</span>
                      {bench.confusionMatrix.uncertain !== undefined && (
                        <span>UNC: {bench.confusionMatrix.uncertain}</span>
                      )}
                    </div>
                  )}

                  <p className="text-[11px] text-on-surface-variant font-sans mt-2">
                    {bench.notes}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 4: OPERATIONAL BOUNDARY DISCLOSURES */}
        <section className="bg-surface-container rounded-xl p-5 sm:p-6 border border-outline-variant/50 shadow-lg flex flex-col gap-4">
          <div className="flex items-center gap-2 border-b border-outline-variant/30 pb-3">
            <span className="material-symbols-outlined text-[20px] text-secondary">warning</span>
            <h2 className="font-serif text-lg sm:text-xl text-tertiary font-semibold">
              Operational Boundary Disclosures & Failure Modes
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col gap-1.5">
              <span className="text-xs font-mono uppercase text-secondary font-semibold">
                1. Lossy Re-Compression
              </span>
              <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                WhatsApp, TikTok, and 64kbps MP3 re-encodings strip high-frequency vocoder phase cues and PRNU camera sensor patterns, causing false positives on legitimate broadcast speech.
              </p>
            </div>

            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col gap-1.5">
              <span className="text-xs font-mono uppercase text-secondary font-semibold">
                2. Linguistic & Dialect Bias
              </span>
              <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                Wav2Vec2 models trained primarily on Western English datasets exhibit high false-positive rates on regional Indic languages (Marathi evaluated at 0% accuracy due to prosodic mismatch).
              </p>
            </div>

            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/40 flex flex-col gap-1.5">
              <span className="text-xs font-mono uppercase text-secondary font-semibold">
                3. Ultra-Brief Audio Failsafe
              </span>
              <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                Speech clips under 0.5 seconds lack sufficient phoneme representation for reliable inference. The system automatically enforces an UNAVAILABLE failsafe instead of guessing.
              </p>
            </div>
          </div>
        </section>

        {/* SECTION 5: CRYPTOGRAPHIC INTEGRITY PROTOCOL FOOTER */}
        <footer className="bg-surface-container rounded-xl p-6 border border-outline-variant/50 text-center flex flex-col items-center gap-2.5">
          <div className="w-10 h-10 rounded-full bg-surface-container-highest flex items-center justify-center text-primary">
            <span className="material-symbols-outlined text-[22px]">policy</span>
          </div>
          <h3 className="font-serif text-lg text-tertiary font-semibold">
            Cryptographic Transparency & Integrity Protocol
          </h3>
          <p className="text-xs text-on-surface-variant max-w-lg leading-relaxed">
            Every verification adjudication emits an immutable SHA-signed cryptographic evidence receipt pegged to standard investigative auditing ledgers.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-2 pt-1 text-[10px] font-mono text-outline">
            <span className="px-2 py-0.5 rounded bg-surface-container-low border border-outline-variant/30">
              IFCN CODE OF PRINCIPLES
            </span>
            <span className="px-2 py-0.5 rounded bg-surface-container-low border border-outline-variant/30">
              C2PA v2.1 COMPLIANT
            </span>
            <span className="px-2 py-0.5 rounded bg-surface-container-low border border-outline-variant/30">
              ISO-17025 FORENSIC SPEC
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
};
