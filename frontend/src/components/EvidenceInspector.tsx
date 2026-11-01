import React, { useMemo } from 'react';
import { EvidenceItem, UnifiedVerificationResponse } from '../types';

/**
 * EvidenceInspector.tsx
 * =====================
 * Expandable evidence cards for a single finding.
 *
 * Every string rendered here comes from the API response. The component never
 * invents sources, URLs, excerpts, or confidence values.
 */
interface Props {
  evidence: EvidenceItem[] | undefined;
  onOpenUrl?: (url: string) => void;
}

const fmtScore = (v: unknown): string => {
  if (typeof v !== 'number' || isNaN(v)) return 'N/A';
  return v > 1 ? `${v.toFixed(1)}%` : `${(v * 100).toFixed(1)}%`;
};

const EvidenceInspector: React.FC<Props> = ({ evidence, onOpenUrl }) => {
  const items: EvidenceItem[] = Array.isArray(evidence) ? evidence : [];

  if (items.length === 0) {
    return (
      <div className="p-4 rounded-lg bg-surface-container-lowest border-dashed border-outline-variant/40 text-xs font-mono text-outline">
        No evidence passages were retrieved for this claim. Nothing was found to support or refute it.
      </div>
    );
  }

  const grouped = useMemo(() => {
    const groups: Record<string, typeof items> = {};
    for (const e of items) {
      const key = e.evidence_class || 'OTHER';
      if (!groups[key]) groups[key] = [];
      groups[key].push(e);
    }
    return groups;
  }, [items]);

  const order = ['SUPPORT', 'CONTRADICTION', 'RELATED_CONTEXT', 'NON_EVIDENCE'];

  return (
    <div className="flex flex-col gap-3">
      {order.map((cls) => {
        const list = grouped[cls];
        if (!list || list.length === 0) return null;
        const isSupport = cls === 'SUPPORT';
        const isContra = cls === 'CONTRADICTION';
        const border = isSupport
          ? 'border-emerald-500/30'
          : isContra
          ? 'border-rose-500/30'
          : 'border-outline-variant/30';
        const title = isSupport ? 'Supporting Evidence' : isContra ? 'Contradicting Evidence' : cls.replace('_', ' ');

        return (
          <div key={cls} className={'rounded-lg border ' + border + ' overflow-hidden bg-surface-container-lowest/40'}>
            <div className={'px-3 py-2 text-[10px] font-mono uppercase tracking-wider font-semibold ' +
              (isSupport ? 'text-emerald-400' : isContra ? 'text-rose-400' : 'text-outline')}>
              {title} ({list.length})
            </div>
            <div className="flex flex-col gap-2 p-3 pt-2">
              {list.map((e, i) => (
                <div key={i} className="p-2.5 rounded bg-surface-container-low border border-outline-variant/20 text-xs">
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <span className="font-mono text-[10px] text-tertiary truncate">
                      {e.source_id || 'unknown-source'}
                    </span>
                    <span className="text-[9px] font-mono text-outline flex-shrink-0">
                      {e.evidence_type || 'evidence'}
                    </span>
                  </div>
                  {e.url && (
                    <a
                      href={e.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={() => onOpenUrl?.(e.url)}
                      className="text-[11px] text-primary hover:underline break-all font-mono"
                    >
                      {e.url}
                    </a>
                  )}
                  <p className="text-[11px] text-on-surface-variant mt-1 leading-relaxed font-sans">
                    {e.chunk}
                  </p>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-2 text-[9px] font-mono text-outline">
                    <span>NLI: {e.nli_label || 'N/A'} ({fmtScore(e.nli_score)})</span>
                    <span>Reliability: {fmtScore(e.source_reliability)}</span>
                    <span>Tier: {e.source_tier ?? 'N/A'}</span>
                    <span>Group: {e.independence_group ?? 'N/A'}</span>
                    <span>Date: {e.date_status || 'N/A'}</span>
                    <span>Score: {fmtScore(e.evidence_score)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default EvidenceInspector;