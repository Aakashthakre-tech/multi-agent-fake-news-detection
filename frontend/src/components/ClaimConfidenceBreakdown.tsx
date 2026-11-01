import React from 'react';
import { UnifiedVerificationResponse } from '../types';
import { formatScore, scoreKindFor } from '../lib/scores';

interface ClaimConfidenceBreakdownProps {
  result: UnifiedVerificationResponse;
}

export const ClaimConfidenceBreakdown: React.FC<ClaimConfidenceBreakdownProps> = ({ result }) => {
  const fact = result.factual_verification || {};
  const synthetic = result.synthetic_media_detection;
  const raw = result.raw || {};
  const evidenceAnalysis = raw.evidence_analysis || [];

  // Calculate real average reliability of sources analyzed
  const avgReliability = React.useMemo(() => {
    if (evidenceAnalysis.length === 0) return null;
    const scores = evidenceAnalysis
      .map((e: any) => e.source_reliability)
      .filter((s: any) => typeof s === 'number');
    if (scores.length === 0) return null;
    return scores.reduce((a: number, b: number) => a + b, 0) / scores.length;
  }, [evidenceAnalysis]);

  // Evidence sufficiency description
  const getSufficiencyBadge = (sufficiency: string | null | undefined) => {
    const s = (sufficiency || 'unknown').toLowerCase();
    if (s.includes('strong_support') || s === 'sufficient') {
      return {
        label: 'SUFFICIENT (STRONG CORROBORATION)',
        desc: 'Multiple independent primary publishers with verified date match.',
        color: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30',
      };
    }
    if (s.includes('support')) {
      return {
        label: 'SUPPORTIVE (CORROBORATED)',
        desc: 'At least two independent publishers corroborate the factual proposition.',
        color: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30',
      };
    }
    if (s.includes('single_source')) {
      return {
        label: 'SINGLE SOURCE ONLY (UNCORROBORATED)',
        desc: 'Only 1 independent publisher found. Minimum 2 required for factual REAL certification.',
        color: 'text-amber-400 bg-amber-500/15 border-amber-500/30',
      };
    }
    if (s.includes('contradiction')) {
      return {
        label: 'CONTRADICTED BY EVIDENCE',
        desc: 'Independent primary reports directly contradict the stated proposition.',
        color: 'text-rose-400 bg-rose-500/15 border-rose-500/30',
      };
    }
    if (s.includes('context')) {
      return {
        label: 'CONTEXT ONLY (INSUFFICIENT)',
        desc: 'Retrieved sources discuss related topics but lack direct factual confirmation.',
        color: 'text-amber-400 bg-amber-500/15 border-amber-500/30',
      };
    }
    return {
      label: 'INSUFFICIENT EVIDENCE',
      desc: 'No independent primary sources retrieved to verify or disprove this proposition.',
      color: 'text-outline bg-surface-container-highest border-outline-variant/40',
    };
  };

  const sufficiencyInfo = getSufficiencyBadge(fact.evidence_sufficiency);

  return (
    <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/50 shadow-md flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-primary">speed</span>
          <span className="text-xs font-mono uppercase tracking-widest text-primary font-semibold">
            Claim-Level Multi-Vector Confidence Matrix
          </span>
        </div>
        <span className="text-[10px] font-mono text-outline">
          STRICT DETERMINISTIC FORMULATIONS
        </span>
      </div>

      {/* Grid of 5 Separate Confidence Vectors */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
        {/* 1. Evidence Sufficiency / Completeness */}
        <div className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/30 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline">
              1. Evidence Sufficiency
            </span>
            <span className="material-symbols-outlined text-[16px] text-primary">data_check</span>
          </div>
          <span
            className={`text-[10px] font-mono px-2 py-1 rounded border font-semibold tracking-wide ${sufficiencyInfo.color}`}
          >
            {sufficiencyInfo.label}
          </span>
          <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
            {sufficiencyInfo.desc}
          </p>
        </div>

        {/* 2. Source Reliability Vector */}
        <div className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/30 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline">
              2. Source Reliability
            </span>
            <span className="material-symbols-outlined text-[16px] text-secondary">verified</span>
          </div>
          <div className="flex items-baseline justify-between">
            <span className="font-serif text-2xl font-bold text-tertiary">
              {avgReliability !== null ? `${(avgReliability * 100).toFixed(1)}%` : 'Not calculated'}
            </span>
            <span className="text-[10px] font-mono text-outline">
              {avgReliability !== null && avgReliability >= 0.8 ? 'TIER 1 DOMAINS' : 'MIXED TIERS'}
            </span>
          </div>
          <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
            {avgReliability !== null
              ? `Average editorial authority score across ${evidenceAnalysis.length} cited source passages.`
              : 'No web sources cited in this run; domain reliability was not evaluated.'}
          </p>
        </div>

        {/* 3. Support & Contradiction Balance */}
        <div className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/30 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline">
              3. Support vs Contradiction
            </span>
            <span className="material-symbols-outlined text-[16px] text-tertiary">balance</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="p-1.5 rounded bg-emerald-500/10 border border-emerald-500/20 flex flex-col">
              <span className="text-base font-mono font-bold text-emerald-400">
                {fact.independent_supporting ?? 0}
              </span>
              <span className="text-[9px] font-mono text-emerald-400/80 uppercase">
                Supporting Groups
              </span>
            </div>
            <div className="p-1.5 rounded bg-rose-500/10 border border-rose-500/20 flex flex-col">
              <span className="text-base font-mono font-bold text-rose-400">
                {fact.independent_contradicting ?? 0}
              </span>
              <span className="text-[9px] font-mono text-rose-400/80 uppercase">
                Contradicting Groups
              </span>
            </div>
          </div>
          <div className="flex justify-between items-center text-[10px] font-mono text-outline pt-0.5">
            <span>Agreement Score:</span>
            <strong className="text-primary">
              {formatScore(fact.source_agreement, scoreKindFor('source_agreement'))}
            </strong>
          </div>
        </div>

        {/* 4. Final Verdict Certainty */}
        <div className="p-3.5 rounded-lg bg-surface-container-low border border-primary/30 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-wider text-primary font-semibold">
              {fact.confidence_basis === 'verification_coverage'
                ? '4. Verification Coverage'
                : '4. Final Verdict Confidence'}
            </span>
            <span className="material-symbols-outlined text-[16px] text-primary">gavel</span>
          </div>
          <div className="flex items-baseline justify-between">
            <span className="font-serif text-3xl font-bold text-primary">
              {fact.confidence !== null && fact.confidence !== undefined
                ? `${(fact.confidence > 1 ? fact.confidence : fact.confidence * 100).toFixed(1)}%`
                : 'Not calculated'}
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-container-high text-primary font-bold">
              {fact.verdict || 'UNVERIFIED'}
            </span>
          </div>
          <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
            {fact.confidence_basis === 'verification_coverage' ? (
              <>
                Share of the retrieved material that could actually be read and
                classified. It is <strong>not</strong> a probability that the claim
                is true or false: UNVERIFIED means the evidence did not settle the
                claim, not that the claim was refuted.
              </>
            ) : (
              'Composite certainty derived from NLI entailment scores, primary publisher count, and source agreement.'
            )}
          </p>
        </div>

        {/* 5. Media Authenticity Confidence (Only applicable when modality warrants it) */}
        <div className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/30 flex flex-col gap-2 sm:col-span-2 lg:col-span-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline">
              5. Media Authenticity
            </span>
            <span className="material-symbols-outlined text-[16px] text-secondary">image_search</span>
          </div>
          {result.modality === 'text' ? (
            <div className="flex flex-col gap-1 my-auto">
              <span className="text-xs font-mono font-semibold text-outline">
                NOT APPLICABLE (TEXT MODALITY)
              </span>
              <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                Media forensic detector (ViT AI-Image / Wav2Vec2) is not invoked for text-only inputs.
              </p>
            </div>
          ) : synthetic?.available ? (
            <div className="flex flex-col gap-2">
              {/* For image modality: show ai_probability and real_probability separately */}
              {result.modality === 'image' && synthetic.probabilities ? (
                <>
                  <div className="flex flex-col gap-0.5">
                    <div className="flex items-baseline justify-between">
                      <span className="text-[10px] font-mono text-outline uppercase">AI-Gen Prob</span>
                      <span className={`font-mono font-bold text-sm ${
                        (() => {
                          const p = synthetic.probabilities?.ai_probability;
                          const pct = p != null ? (p > 1 ? p : p * 100) : null;
                          return pct != null && pct >= 73 ? 'text-rose-400' : pct != null && pct <= 27 ? 'text-emerald-400' : 'text-amber-400';
                        })()
                      }`}>
                        {(() => {
                          const p = synthetic.probabilities?.ai_probability;
                          const pct = p != null ? (p > 1 ? p : p * 100) : null;
                          return pct != null ? `${pct.toFixed(1)}%` : 'N/A';
                        })()}
                      </span>
                    </div>
                    <div className="flex items-baseline justify-between">
                      <span className="text-[10px] font-mono text-outline uppercase">Natural Photo Prob</span>
                      <span className={`font-mono font-bold text-sm ${
                        (() => {
                          const p = synthetic.probabilities?.real_probability;
                          const pct = p != null ? (p > 1 ? p : p * 100) : null;
                          return pct != null && pct >= 73 ? 'text-emerald-400' : pct != null && pct <= 27 ? 'text-rose-400' : 'text-amber-400';
                        })()
                      }`}>
                        {(() => {
                          const p = synthetic.probabilities?.real_probability;
                          const pct = p != null ? (p > 1 ? p : p * 100) : null;
                          return pct != null ? `${pct.toFixed(1)}%` : 'N/A';
                        })()}
                      </span>
                    </div>
                    <div className="flex items-baseline justify-between">
                      <span className="text-[10px] font-mono text-outline uppercase">Tampering</span>
                      <span className="font-mono font-bold text-[10px] text-amber-400">
                        {synthetic.manipulation_status || 'NOT_ASSESSED'}
                      </span>
                    </div>
                  </div>
                  <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                    Model: {synthetic.model || 'not reported'}. Signal: <strong className="uppercase">{synthetic.signal || 'UNAVAILABLE'}</strong>.
                  </p>
                </>
              ) : (
                <div className="flex flex-col gap-1">
                  <div className="flex items-baseline justify-between">
                    <span className="font-serif text-2xl font-bold text-secondary">
                      {synthetic.confidence !== null ? `${(synthetic.confidence * 100).toFixed(1)}%` : 'Evaluated'}
                    </span>
                    <span className="text-[10px] font-mono text-secondary font-bold uppercase">
                      {synthetic.signal || 'UNAVAILABLE'}
                    </span>
                  </div>
                  <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                    Model: {synthetic.model || 'not reported'}. Evaluates whether pixels or audio spectra exhibit synthetic generation signatures.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col gap-1 my-auto">
              <span className="text-xs font-mono font-semibold text-amber-400">
                NOT CALCULATED
              </span>
              <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                No active synthetic media classifier was configured for this input format.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
