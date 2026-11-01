import React, { useState } from 'react';
import { ClaimRecord } from '../types';

/**
 * ClaimBreakdownCard.tsx
 * ======================
 * Expandable card for a single claim. Renders only what the backend returned.
 * No fabricated verdicts, sources, or confidence values.
 */
interface Props {
  claim: ClaimRecord;
  index: number;
}

const isSafeUrl = (url?: string | null): boolean => {
  if (!url) return false;
  const candidate = url.trim();
  return (
    (candidate.startsWith('http://') || candidate.startsWith('https://')) &&
    !/\s/.test(candidate)
  );
};

const verdictColor = (v?: string | null) => {
  const s = (v || '').toUpperCase();
  if (s === 'SUPPORTED' || s === 'REAL' || s === 'TRUE') return 'text-emerald-400';
  if (s === 'CONTRADICTED' || s === 'FAKE' || s === 'FALSE') return 'text-rose-400';
  if (s === 'MISLEADING') return 'text-amber-400';
  return 'text-outline';
};

const ClaimBreakdownCard: React.FC<Props> = ({ claim, index }) => {
  const [expanded, setExpanded] = useState(false);

  const verdict = claim.claim_verdict || claim.verdict || null;
  const supporting = claim.supporting_evidence || [];
  const contradicting = claim.contradicting_evidence || [];
  const hasEvidence = supporting.length > 0 || contradicting.length > 0;

  const fmtScore = (v?: number | null) => {
    if (v == null || isNaN(v)) return 'N/A';
    return v > 1 ? `${v.toFixed(1)}%` : `${(v * 100).toFixed(1)}%`;
  };

  const timestampRange = (): string | null => {
    const t = claim.timestamps;
    if (!t || !Array.isArray(t) || t.length !== 2) return null;
    return `${t[0]}s - ${t[1]}s`;
  };

  return (
    <div className={'rounded-lg border border-outline-variant/30 overflow-hidden ' + (expanded ? 'bg-surface-container-high' : 'bg-surface-container-low')}>
      <div
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between gap-3 p-3 cursor-pointer"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[10px] font-mono text-tertiary flex-shrink-0">#{index + 1}</span>
          <span className={'text-xs font-mono font-semibold ' + verdictColor(verdict)}>
            {(verdict || 'UNVERIFIED').toUpperCase()}
          </span>
          {claim.source_modality && (
            <span className="text-[9px] font-mono text-outline px-1.5 py-0.5 rounded bg-surface-container-high">
              {claim.source_modality}
            </span>
          )}
        </div>
        <span className="material-symbols-outlined text-[16px] text-outline flex-shrink-0">
          {expanded ? 'expand_less' : 'expand_more'}
        </span>
      </div>

      <div className="px-3 pb-3">
        <p className="text-xs text-on-surface font-sans mt-1 leading-relaxed">
          {claim.claim_text || claim.claim || claim.original_text || 'No claim text available.'}
        </p>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-2 text-[10px] font-mono text-outline">
          {claim.priority_score != null && <span>Priority: {claim.priority_score.toFixed(2)}</span>}
          {claim.confidence != null && <span>Confidence: {fmtScore(claim.confidence)}</span>}
          {claim.evidence_sufficiency && <span>Sufficiency: {claim.evidence_sufficiency}</span>}
          {claim.selection_reason && <span className="text-on-surface-variant italic">{claim.selection_reason}</span>}
        </div>

        {timestampRange() && (
          <div className="mt-2 text-[10px] font-mono text-outline">
            Timestamp: {timestampRange()}
            {claim.frame_references && claim.frame_references.length > 0 && (
              <>  |  Frames: {claim.frame_references.join(', ')}</>
            )}
          </div>
        )}

        {expanded && (
          <div className="mt-3 pt-3 border-t border-outline-variant/20 flex flex-col gap-2">
            {claim.selection_reason && (
              <div className="p-2 rounded bg-surface-container-lowest text-[11px] text-on-surface-variant font-sans">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">Selection Reason</span>
                {claim.selection_reason}
              </div>
            )}

            {claim.reasoning && (
              <div className="p-2 rounded bg-surface-container-lowest text-[11px] text-on-surface-variant font-sans">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">Reasoning</span>
                {claim.reasoning}
              </div>
            )}

            {hasEvidence ? (
              <div className="flex flex-col gap-1.5">
                {(['supporting_evidence', 'contradicting_evidence'] as const).map((key) => {
                  const items = claim[key] || [];
                  if (!items.length) return null;
                  return (
                    <div key={key}>
                      <span className="text-[9px] font-mono uppercase text-outline block mb-1">
                        {key === 'supporting_evidence' ? 'Supporting evidence' : 'Contradicting evidence'}
                      </span>
                      {items.map((ev: any, j: number) => (
                        <div
                          key={j}
                          className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20 mb-1"
                        >
                          <p className="text-[10px] font-mono text-on-surface">
                            {ev?.domain || 'source'}
                            {ev?.source_tier ? ` | tier: ${ev.source_tier}` : ''}
                            {ev?.date_status ? ` | date: ${ev.date_status}` : ''}
                          </p>
                          {ev?.chunk && (
                            <p className="text-[10px] text-on-surface-variant font-sans leading-relaxed mt-1">
                              {ev.chunk}
                            </p>
                          )}
                          {isSafeUrl(ev?.url) && (
                            <a
                              href={ev.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                              className="inline-flex items-center gap-1 text-[9px] font-mono text-primary hover:underline mt-1"
                            >
                              <span className="material-symbols-outlined text-[11px]">open_in_new</span>
                              Visit source
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-dashed border-outline-variant/40 text-[11px] font-mono text-outline">
                No evidence was retrieved for this claim. Absence of evidence does not mean the claim is false.
              </div>
            )}

            {claim.limitations && claim.limitations.length > 0 && (
              <div className="p-2 rounded bg-surface-container-lowest text-[11px] text-on-surface-variant font-sans">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  Uncertainties &amp; limitations
                </span>
                <ul>
                  {claim.limitations.map((l, j) => (
                    <li key={j}>• {l}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ClaimBreakdownCard;
