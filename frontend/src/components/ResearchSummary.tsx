import React, { useState } from 'react';
import { UnifiedVerificationResponse } from '../types';

/**
 * ResearchSummary.tsx
 * ===================
 * Evidence-grounded research summary rendered from the actual API response.
 *
 * The summary is derived deterministically from the factual-verification
 * fields the backend already produced. It never invents sources, quotes,
 * statistics, or conclusions that are not traceable to the response payload.
 */
interface Props {
  result: UnifiedVerificationResponse;
}

const pct = (v?: number | null): string => {
  if (v == null || isNaN(v)) return 'N/A';
  return v > 1 ? `${v.toFixed(1)}%` : `${(v * 100).toFixed(1)}%`;
};

const ResearchSummary: React.FC<Props> = ({ result }) => {
  const [expanded, setExpanded] = useState(true);
  const fact = result.factual_verification || {};
  const raw = result.raw || {};
  const finalV = raw.final_verdict || {};
  const evidence = raw.evidence_analysis || [];
  const sources = raw.source_data || [];

  const supporting = evidence.filter((e: any) => e.evidence_class === 'SUPPORT');
  const contradicting = evidence.filter((e: any) => e.evidence_class === 'CONTRADICTION');
  const context = evidence.filter((e: any) => e.evidence_class === 'RELATED_CONTEXT');

  const verdict = fact.verdict || 'UNVERIFIED';
  const explanation =
    fact.deterministic_gate_reason ||
    fact.reasoning ||
    finalV.permitted_reason ||
    'The evidence layer completed analysis. No narrative rationale was supplied.';

  const sufficiency = (fact.evidence_sufficiency || '').toLowerCase();
  const hasSupport = supporting.length > 0;
  const hasContra = contradicting.length > 0;

  // Deterministic summary lines grounded in the actual response.
  const lines: string[] = [];
  if (hasSupport && hasContra) {
    lines.push(
      `The investigation found ${supporting.length} supporting passage(s) and ${contradicting.length} contradicting passage(s) ` +
      `across ${sources.length} source(s). Independent sources disagree, so the claim is genuinely contested.`
    );
  } else if (hasSupport) {
    lines.push(
      `The investigation found ${supporting.length} supporting passage(s) across ${sources.length} source(s). ` +
      `No contradicting evidence was retrieved.`
    );
  } else if (hasContra) {
    lines.push(
      `The investigation found ${contradicting.length} contradicting passage(s) across ${sources.length} source(s). ` +
      `No supporting evidence was retrieved.`
    );
  } else if (sources.length > 0) {
    lines.push(
      `${sources.length} source(s) were retrieved, but none passed the evidence-layer filters as direct support or contradiction. ` +
      `The retrieved material is context only.`
    );
  } else {
    lines.push('No sources were retrieved for this claim. There is nothing to support or refute it.');
  }

  if (sufficiency.includes('single_source')) {
    lines.push('Only one independent publisher corroborated the claim; two or more independent primary publishers are required for a confident REAL verdict.');
  } else if (sufficiency.includes('strong_support') || sufficiency.includes('support')) {
    lines.push(`At least ${fact.independent_supporting || 0} independent publisher(s) directly support the claim.`);
  } else if (sufficiency.includes('contradiction')) {
    lines.push(`${fact.independent_contradicting || 0} independent publisher(s) directly contradict the claim.`);
  }

  if (context.length > 0) {
    lines.push(`${context.length} source passage(s) discussed related topics but did not establish the claimed event with the claimed date and location.`);
  }

  if (evidence.some((e: any) => e.evidence_type === 'search_snippet')) {
    lines.push('At least one source is a search-snippet fallback rather than a full retrieved article. Snippet-only evidence can never establish or refute a claim.');
  }

  if (verdict === 'UNVERIFIED') {
    lines.push('A verdict of UNVERIFIED means independent evidence was not found. It does NOT prove the claim false.');
  }

  return (
    <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/50 shadow-md flex flex-col gap-3">
      <div
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-primary">summarize</span>
          <span className="text-xs font-mono uppercase tracking-widest text-primary font-semibold">
            Executive Research Summary
          </span>
        </div>
        <span className="material-symbols-outlined text-[16px] text-outline">
          {expanded ? 'expand_less' : 'expand_more'}
        </span>
      </div>

      {expanded && (
        <div className="flex flex-col gap-3 pt-2 border-t border-outline-variant/30">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
            <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
              <div className="text-base font-mono font-bold text-emerald-400">{supporting.length}</div>
              <div className="text-[9px] font-mono text-outline uppercase">Supporting</div>
            </div>
            <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
              <div className="text-base font-mono font-bold text-rose-400">{contradicting.length}</div>
              <div className="text-[9px] font-mono text-outline uppercase">Contradicting</div>
            </div>
            <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
              <div className="text-base font-mono font-bold text-on-surface">{sources.length}</div>
              <div className="text-[9px] font-mono text-outline uppercase">Sources</div>
            </div>
            <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
              <div className="text-base font-mono font-bold text-primary">{pct(fact.confidence)}</div>
              <div className="text-[9px] font-mono text-outline uppercase">Confidence</div>
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            {lines.map((line, i) => (
              <p key={i} className="text-xs text-on-surface font-sans leading-relaxed">
                {line}
              </p>
            ))}
          </div>

          <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
            <span className="text-[10px] font-mono uppercase text-outline block mb-0.5">Adjudication Explanation</span>
            <p className="text-xs text-on-surface-variant font-sans leading-relaxed">{explanation}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResearchSummary;