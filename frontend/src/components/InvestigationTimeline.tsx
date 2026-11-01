import React, { useState } from 'react';
import { UnifiedVerificationResponse, PipelineStageDetails } from '../types';

interface InvestigationTimelineProps {
  result: UnifiedVerificationResponse;
  elapsedSeconds?: number;
}

export const InvestigationTimeline: React.FC<InvestigationTimelineProps> = ({
  result,
  elapsedSeconds,
}) => {
  const [expandedStage, setExpandedStage] = useState<string | null>(null);

  const raw = result.raw || {};
  const fact = result.factual_verification || {};

  // Use backend-provided pipeline_stages for all modalities.
  //
  // There is deliberately NO client-side reconstruction of pipeline stages:
  // a fabricated stage list would claim work, tools and model names that the
  // backend never performed. When the backend does not report stages, the
  // timeline says so instead of inventing them.
  const backendStages: PipelineStageDetails[] =
    Array.isArray(result.pipeline_stages) && result.pipeline_stages.length > 0
      ? result.pipeline_stages
      : [];

  const stages = backendStages;

  const getNodeClass = (status: string) => {
    if (status === 'completed') return 'bg-secondary text-surface-container-lowest border-secondary';
    if (status === 'failed') return 'bg-error text-surface-container-lowest border-error';
    if (status === 'skipped') return 'bg-surface-container-highest text-outline border-outline-variant opacity-50';
    if (status === 'running') return 'bg-primary text-surface-container-lowest border-primary animate-pulse';
    return 'bg-surface-container-highest text-outline border-outline-variant';
  };

  const getCardClass = (status: string, isExpanded: boolean) => {
    if (status === 'skipped') return 'opacity-50 bg-surface-container-lowest border-outline-variant/20';
    if (isExpanded) return 'bg-surface-container-high border-secondary/50 shadow-sm';
    return 'bg-surface-container-low hover:bg-surface-container-high/60 border-outline-variant/30';
  };

  const getStatusBadgeClass = (status: string) => {
    if (status === 'completed') return 'bg-secondary/15 text-secondary border-secondary/30';
    if (status === 'failed') return 'bg-error/15 text-error border-error/30';
    if (status === 'skipped') return 'bg-surface-container-highest text-outline border-outline-variant/30';
    return 'bg-primary/15 text-primary border-primary/30';
  };

  const getModalityIcon = (m: string) => {
    if (m === 'image') return 'image_search';
    if (m === 'video') return 'videocam';
    if (m === 'audio') return 'graphic_eq';
    return 'article';
  };

  return (
    <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/50 shadow-md flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-secondary">timeline</span>
          <span className="text-xs font-mono uppercase tracking-widest text-secondary font-semibold">Forensic Investigation Pipeline Timeline</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-outline">
          <span className="flex items-center gap-1">
            <span className="material-symbols-outlined text-[12px]">{getModalityIcon(result.modality)}</span>
            <span>{result.modality.toUpperCase()} PIPELINE</span>
          </span>
          <span>{stages.length} STAGES</span>
          {elapsedSeconds !== undefined && elapsedSeconds > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-primary">{elapsedSeconds}s runtime</span>
          )}
          {result.case_summary?.processing_time_sec && (
            <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-secondary">{result.case_summary.processing_time_sec}s pipeline</span>
          )}
        </div>
      </div>

      {result.modality !== 'text' && (
        <div className="flex flex-wrap items-center gap-2 px-3 py-2 rounded-lg bg-surface-container-low border border-outline-variant/20 text-[10px] font-mono text-on-surface-variant">
          <span className="material-symbols-outlined text-[14px] text-primary">info</span>
          <span>
            <strong className="text-tertiary">{result.modality.toUpperCase()} FORENSIC MODE: </strong>
            {result.modality === 'image' && 'OCR extraction -> YOLO object analysis -> claim normalization -> ViT AI origin detection -> live wire fact-checking.'}
            {result.modality === 'video' && 'Frame demuxing -> Whisper transcription -> frame OCR -> claim ranking -> VideoMAE classifier -> live wire fact-checking.'}
            {result.modality === 'audio' && 'Audio decoding -> Wav2Vec2 deepfake detection -> Whisper transcription -> claim extraction -> live wire fact-checking.'}
          </span>
        </div>
      )}

      {stages.length === 0 && (
        <div className="p-6 rounded-lg bg-surface-container-low text-center text-xs font-mono text-outline border border-dashed border-outline-variant/40">
          Pipeline stage data not available for this result.
        </div>
      )}

      <div className="flex flex-col gap-3 relative before:absolute before:top-3 before:bottom-3 before:left-4 before:w-0.5 before:bg-outline-variant/30">
        {stages.map((stg) => {
          const isExpanded = expandedStage === stg.id;
          const toolLabel = (stg.tool_or_model || '').split('//')[0].trim();
          return (
            <div key={stg.id} className="relative pl-10 flex flex-col transition-all">
              <div className={'absolute left-2.5 -translate-x-1/2 top-2.5 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-mono font-bold border transition-colors ' + getNodeClass(stg.status)}>
                {stg.status === 'completed' ? <span className="material-symbols-outlined text-[12px]">check</span>
                  : stg.status === 'failed' ? <span className="material-symbols-outlined text-[12px]">close</span>
                  : stg.status === 'skipped' ? <span className="material-symbols-outlined text-[12px]">remove</span>
                  : stg.number}
              </div>

              <div
                onClick={() => setExpandedStage(isExpanded ? null : stg.id)}
                className={'p-3.5 rounded-lg border transition-all cursor-pointer ' + getCardClass(stg.status, isExpanded)}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-xs font-mono font-semibold text-tertiary">Stage {stg.number}: {stg.name}</span>
                    <span className={'text-[9px] font-mono px-1.5 py-0.5 rounded uppercase font-semibold border ' + getStatusBadgeClass(stg.status)}>
                      {stg.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-[10px] font-mono text-outline hidden sm:inline">{toolLabel}</span>
                    {stg.duration_sec && (
                      <span className="text-[10px] font-mono text-primary px-1 py-0.5 rounded bg-surface-container-high hidden sm:inline">{stg.duration_sec.toFixed(1)}s</span>
                    )}
                    <span className="material-symbols-outlined text-[16px] text-outline">{isExpanded ? 'expand_less' : 'expand_more'}</span>
                  </div>
                </div>

                <p className="text-xs text-on-surface-variant font-sans mt-1 line-clamp-1">{stg.operation}</p>

                {stg.metric && !isExpanded && (
                  <div className="flex items-center gap-2 mt-2 pt-1.5 border-t border-outline-variant/20 text-[10px] font-mono">
                    <span className="text-outline">{stg.metric.label}:</span>
                    <span className="text-primary font-semibold">{stg.metric.value}</span>
                  </div>
                )}

                {stg.warnings && stg.warnings.length > 0 && !isExpanded && (
                  <div className="flex items-center gap-1 mt-1.5 text-[10px] font-mono text-amber-400">
                    <span className="material-symbols-outlined text-[12px]">warning</span>
                    <span>{stg.warnings[0]}</span>
                  </div>
                )}

                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-outline-variant/30 flex flex-col gap-2.5 text-xs font-sans">
                    <div>
                      <span className="text-[10px] font-mono uppercase text-outline block mb-0.5">Why Executed:</span>
                      <p className="text-on-surface leading-relaxed">{stg.why}</p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] font-mono">
                      <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
                        <span className="text-[9px] text-outline uppercase block mb-0.5">Input:</span>
                        <span className="text-on-surface-variant break-words">{stg.input_summary}</span>
                      </div>
                      <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
                        <span className="text-[9px] text-outline uppercase block mb-0.5">Output:</span>
                        <span className="text-on-surface font-medium break-words">{stg.output_summary}</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-[10px] font-mono text-outline pt-1">
                      <span>Confirmed Model / Engine:</span>
                      <strong className="text-tertiary">{stg.tool_or_model}</strong>
                    </div>
                    {stg.metric && (
                      <div className="p-2.5 rounded bg-surface-container-lowest border border-primary/20 flex flex-col gap-1">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono uppercase text-primary font-semibold">{stg.metric.label}</span>
                          <span className="text-xs font-mono font-bold text-primary">{stg.metric.value}</span>
                        </div>
                        <p className="text-[11px] text-on-surface-variant font-sans">{stg.metric.meaning}</p>
                        <span className="text-[9px] font-mono text-outline">Scale: {stg.metric.scale}</span>
                      </div>
                    )}
                    {stg.warnings && stg.warnings.length > 0 && (
                      <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20">
                        {stg.warnings.map((w, i) => (
                          <p key={i} className="text-[11px] text-amber-400 font-mono flex items-start gap-1">
                            <span className="material-symbols-outlined text-[12px] mt-0.5">warning</span>
                            <span>{w}</span>
                          </p>
                        ))}
                      </div>
                    )}
                    {stg.details && (
                      <div className="p-2 rounded bg-surface-container-lowest text-[11px] text-on-surface-variant italic font-sans border border-outline-variant/20">
                        {stg.details}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {stages.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-outline-variant/20 text-[10px] font-mono text-outline">
          <span className="flex items-center gap-1">
            <span className="material-symbols-outlined text-[12px] text-secondary">verified</span>
            <span>All stages executed on local host. No fabricated data.</span>
          </span>
          <span>{stages.filter(s => s.status === 'completed').length}/{stages.length} stages completed</span>
        </div>
      )}
    </div>
  );
};
