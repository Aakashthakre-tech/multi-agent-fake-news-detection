import React, { useEffect, useRef } from 'react';
import { PipelineProgressEvent } from '../services/api';

// ─── Stage metadata ──────────────────────────────────────────────────────────

const STAGES = [
  { id: 'ingestion',    label: 'Ingestion',      icon: 'input'               },
  { id: 'research',     label: 'Web Research',   icon: 'travel_explore'      },
  { id: 'evidence',     label: 'Evidence',       icon: 'collections_bookmark'},
  { id: 'analysis',     label: 'Analysis',       icon: 'biotech'             },
  { id: 'fusion',       label: 'Fusion',         icon: 'hub'                 },
  { id: 'brief',        label: 'Brief',          icon: 'summarize'           },
  { id: 'verify_agent', label: 'Verifier Agent', icon: 'smart_toy'           },
  { id: 'critic',       label: 'Critic Review',  icon: 'rate_review'         },
  { id: 'verdict',      label: 'Verdict',        icon: 'gavel'               },
];

const STAGE_COLORS: Record<string, string> = {
  ingestion:    'text-primary',
  research:     'text-secondary',
  evidence:     'text-tertiary',
  analysis:     'text-secondary',
  fusion:       'text-primary',
  brief:        'text-tertiary',
  verify_agent: 'text-secondary',
  critic:       'text-primary',
  verdict:      'text-tertiary',
};

// ─── Log entry ───────────────────────────────────────────────────────────────

export interface LogEntry {
  id: string;
  stage: string;
  label: string;
  icon: string;
  detail: string;
  step: number;
  total_steps: number;
  ts: number; // performance.now()
}

// ─── Component Props ──────────────────────────────────────────────────────────

interface LiveInvestigationPanelProps {
  /** All log entries accumulated so far */
  logs: LogEntry[];
  /** Latest / current stage id */
  currentStage: string;
  /** Current step number (1-indexed) */
  currentStep: number;
  /** Total steps */
  totalSteps: number;
  /** Elapsed seconds */
  elapsedSeconds: number;
  /** Whether the pipeline is still running */
  isRunning: boolean;
  /** Whether the pipeline finished with an error */
  hasError: boolean;
}

// ─── Main Component ───────────────────────────────────────────────────────────

export const LiveInvestigationPanel: React.FC<LiveInvestigationPanelProps> = ({
  logs,
  currentStage,
  currentStep,
  totalSteps,
  elapsedSeconds,
  isRunning,
  hasError,
}) => {
  const logScrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log window to bottom on new entries
  useEffect(() => {
    const el = logScrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs.length]);

  const progressPct = totalSteps > 0 ? Math.min(100, (currentStep / totalSteps) * 100) : 0;
  const currentStageInfo = STAGES.find((s) => s.id === currentStage) ?? STAGES[0];

  return (
    <div className="flex flex-col gap-4 animate-in fade-in duration-300">
      {/* ── Header bar ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              hasError
                ? 'bg-error'
                : isRunning
                ? 'bg-secondary animate-pulse'
                : 'bg-primary'
            }`}
          />
          <span className="text-[11px] font-mono uppercase tracking-widest font-semibold text-secondary">
            {hasError
              ? 'Pipeline Fault'
              : isRunning
              ? 'Live Investigation'
              : 'Investigation Complete'}
          </span>
        </div>
        <span className="text-[10px] font-mono text-outline">
          {elapsedSeconds}s elapsed
        </span>
      </div>

      {/* ── Stage progress rail ─────────────────────────────────────────── */}
      <div className="bg-surface-container-low rounded-xl p-4 border border-outline-variant/40 shadow-md">
        {/* Stage label */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className={`material-symbols-outlined text-[18px] ${STAGE_COLORS[currentStage] ?? 'text-primary'}`}>
              {currentStageInfo.icon}
            </span>
            <span className={`text-xs font-mono font-semibold ${STAGE_COLORS[currentStage] ?? 'text-primary'}`}>
              {currentStageInfo.label}
            </span>
          </div>
          <span className="text-[10px] font-mono text-outline">
            Step {currentStep}/{totalSteps}
          </span>
        </div>

        {/* Progress bar */}
        <div className="w-full h-1.5 bg-surface-container-lowest rounded-full overflow-hidden mb-3">
          <div
            className="h-full bg-gradient-to-r from-primary via-secondary to-tertiary rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progressPct}%` }}
          />
        </div>

        {/* Stage dots */}
        <div className="flex items-center justify-between gap-1">
          {STAGES.map((stage, idx) => {
            const stageStep = idx + 1;
            const isActive = stage.id === currentStage;
            const isDone = stageStep < currentStep || (!isRunning && !hasError && stageStep <= currentStep);
            return (
              <div
                key={stage.id}
                className="flex flex-col items-center gap-1 flex-1"
                title={stage.label}
              >
                <div
                  className={`w-5 h-5 rounded-full flex items-center justify-center transition-all duration-300 ${
                    isActive && isRunning
                      ? 'bg-secondary ring-2 ring-secondary/40 scale-110'
                      : isDone
                      ? 'bg-primary/70'
                      : hasError && isActive
                      ? 'bg-error'
                      : 'bg-surface-container-high'
                  }`}
                >
                  {isDone && !isActive ? (
                    <span className="material-symbols-outlined text-[10px] text-on-primary">
                      check
                    </span>
                  ) : isActive && isRunning ? (
                    <span className="material-symbols-outlined text-[10px] text-on-secondary animate-spin">
                      progress_activity
                    </span>
                  ) : (
                    <span className="material-symbols-outlined text-[10px] text-outline">
                      {stage.icon}
                    </span>
                  )}
                </div>
                <span className="text-[8px] font-mono text-outline hidden sm:block truncate w-full text-center">
                  {stage.label.split(' ')[0]}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Live log stream ─────────────────────────────────────────────── */}
      <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/40 overflow-hidden shadow-md">
        {/* Log header */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-outline-variant/30 bg-surface-container-low">
          <span className="text-[10px] font-mono uppercase tracking-wider text-tertiary font-semibold flex items-center gap-1.5">
            <span className="material-symbols-outlined text-[14px]">terminal</span>
            Pipeline Event Stream
          </span>
          <span className="text-[9px] font-mono text-outline">
            {logs.length} events
          </span>
        </div>

        {/* Scrollable log body */}
        <div
          ref={logScrollRef}
          className="overflow-y-auto font-mono text-[11px] leading-relaxed"
          style={{ maxHeight: '280px' }}
        >
          {logs.length === 0 ? (
            <div className="p-4 text-outline text-center text-[11px] font-mono">
              Awaiting pipeline events…
            </div>
          ) : (
            <div className="flex flex-col divide-y divide-outline-variant/10">
              {logs.map((entry, i) => {
                const isLatest = i === logs.length - 1 && isRunning;
                const color = STAGE_COLORS[entry.stage] ?? 'text-outline';
                return (
                  <div
                    key={entry.id}
                    className={`flex items-start gap-2.5 px-3 py-2 transition-colors ${
                      isLatest ? 'bg-surface-container/60' : ''
                    }`}
                  >
                    {/* Step badge */}
                    <span className="flex-shrink-0 w-4 text-[9px] text-outline mt-0.5 text-right">
                      {entry.step}
                    </span>

                    {/* Icon */}
                    <span className={`material-symbols-outlined text-[14px] flex-shrink-0 mt-0.5 ${color}`}>
                      {entry.icon}
                    </span>

                    {/* Content */}
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className={`font-semibold ${color} truncate`}>
                        {entry.label}
                      </span>
                      {entry.detail && entry.detail !== entry.label && (
                        <span className="text-outline text-[10px] truncate mt-0.5">
                          {entry.detail}
                        </span>
                      )}
                    </div>

                    {/* Live spinner for latest */}
                    {isLatest && (
                      <span className="material-symbols-outlined text-[14px] text-secondary animate-spin flex-shrink-0">
                        progress_activity
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Stage summary badges ────────────────────────────────────────── */}
      {logs.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {[...new Set(logs.map((l) => l.stage))].map((stageId) => {
            const meta = STAGES.find((s) => s.id === stageId);
            if (!meta) return null;
            const stageCount = logs.filter((l) => l.stage === stageId).length;
            return (
              <div
                key={stageId}
                className="flex items-center gap-1 bg-surface-container-high px-2 py-0.5 rounded-full border border-outline-variant/30"
              >
                <span className={`material-symbols-outlined text-[12px] ${STAGE_COLORS[stageId] ?? 'text-outline'}`}>
                  {meta.icon}
                </span>
                <span className="text-[10px] font-mono text-on-surface-variant">
                  {meta.label.split(' ')[0]}
                </span>
                <span className="text-[9px] font-mono text-outline">
                  ×{stageCount}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
