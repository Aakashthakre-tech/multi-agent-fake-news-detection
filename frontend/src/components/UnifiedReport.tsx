// src/components/UnifiedReport.tsx
// Compact, single-card verification result. Replaces the long forensic
// dossier layout with one professional summary block.
import React from 'react';
import { UnifiedVerificationResponse } from '../types';

interface UnifiedReportProps {
  result: UnifiedVerificationResponse;
  modality: string;
  claimText: string;
  selectedFile: File | null;
}

const VERDICT_STYLE: Record<string, { label: string; badge: string; dot: string; accent: string }> = {
  REAL: {
    label: 'REAL',
    badge: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
    dot: 'bg-emerald-400',
    accent: 'text-emerald-400',
  },
  FAKE: {
    label: 'FAKE',
    badge: 'bg-rose-500/15 border-rose-500/30 text-rose-400',
    dot: 'bg-rose-400',
    accent: 'text-rose-400',
  },
  MISLEADING: {
    label: 'MISLEADING',
    badge: 'bg-amber-500/15 border-amber-500/30 text-amber-400',
    dot: 'bg-amber-400',
    accent: 'text-amber-400',
  },
  UNVERIFIED: {
    label: 'UNVERIFIED',
    badge: 'bg-slate-500/15 border-slate-500/30 text-slate-400',
    dot: 'bg-slate-400',
    accent: 'text-slate-400',
  },
};

const EVIDENCE_STATUS: Record<string, { label: string; color: string }> = {
  SUFFICIENT: { label: 'SUFFICIENT', color: 'text-emerald-400' },
  LIMITED: { label: 'LIMITED', color: 'text-amber-400' },
  INSUFFICIENT: { label: 'INSUFFICIENT', color: 'text-slate-400' },
};

/**
 * Generate a plain-English explanation of why the verdict was reached,
 * based only on the validated evidence and the deterministic verdict.
 * Never hardcodes a specific claim or expected outcome.
 */
function generateFinalVerdictExplanation(
  verdict: string,
  independentSupporting: number,
  independentContradicting: number,
  evidenceSufficiency: string,
  reasoning?: string,
  primaryClaim?: string
): string {
  if (reasoning && reasoning.trim().length > 0) {
    return reasoning
      .split(/(?<=[.!?])\s+/)
      .filter((s) => s.trim().length > 0)
      .slice(0, 3)
      .join(' ');
  }

  switch (verdict) {
    case 'REAL':
      return 'Multiple independent, reliable sources directly support this claim. Their reports agree on the main event details, and no qualifying contradictory evidence was found.';
    case 'FAKE':
      return 'Multiple independent, reliable sources directly contradict this claim. The available evidence refutes the stated facts.';
    case 'MISLEADING':
      return 'Independent sources confirm the core event but identify material distortions or omissions in how it is presented.';
    case 'UNVERIFIED':
      if (independentSupporting === 0 && independentContradicting === 0) {
        return 'The available evidence is insufficient to confirm or refute this claim. No independent, reliable sources were found that directly establish or refute this claim. The available material is related context only.';
      }
      if (independentSupporting === 1 && independentContradicting === 0) {
        return 'One independent, reliable source supports this claim. TruthLensAI requires at least two independent publishers to confirm a claim as REAL.';
      }
      if (independentSupporting >= 2 && independentContradicting === 0) {
        return 'Multiple sources mention this claim, but they do not meet the evidence requirements for a confirmed verdict.';
      }
      if (independentContradicting >= 1) {
        return 'Independent sources present conflicting accounts of this claim. The evidence is too divided to reach a settled verdict.';
      }
      return 'The available evidence does not establish this claim as true or false.';
    default:
      return 'The available evidence does not establish this claim as true or false.';
  }
}

/**
 * Short plain-English verdict rationale for the "Why This Verdict" line.
 * Falls back to a generic statement when the backend did not supply one.
 */
function generateVerdictReasoning(
  verdict: string,
  independentSupporting: number,
  independentContradicting: number,
  evidenceSufficiency: string
): string {
  switch (verdict) {
    case 'REAL':
      return 'Two or more independent publishers directly support this claim with retrieved article text, meeting the evidence policy for a settled verdict.';
    case 'FAKE':
      return 'Multiple independent publishers directly contradict this claim with retrieved article text, meeting the evidence policy for a settled verdict.';
    case 'MISLEADING':
      return 'Independent sources confirm the core event but identify material distortions or omissions in how it is presented.';
    case 'UNVERIFIED':
      if (independentSupporting === 0 && independentContradicting === 0) {
        return 'No independent publisher directly establishes or refutes this claim; the available evidence is related context only.';
      }
      if (independentSupporting === 1 && independentContradicting === 0) {
        return 'Only one independent publisher supports this claim; corroboration from a second independent publisher is required.';
      }
      if (independentContradicting >= 1) {
        return 'Independent sources present conflicting accounts; the evidence is too divided for a settled verdict.';
      }
      return 'The available evidence does not establish this claim as true or false.';
    default:
      return 'The available evidence does not establish this claim as true or false.';
  }
}

/**
 * Generate a plain-English final decision summary that explains the
 * relationship between the Google Fact Check result and the TruthLensAI
 * verdict.
 */
function generateFinalDecisionSummary(
  verdict: string,
  independentSupporting: number,
  independentContradicting: number,
  googleFactCheck: any,
  evidenceSufficiency: string,
  reasoning?: string
): string {
  const gc = googleFactCheck;
  const hasGoogleResult = gc && gc.status === 'success' && gc.claims && gc.claims.length > 0;
  const hasRelevantReview = hasGoogleResult && gc.claims.some((c: any) =>
    c.claimReview && c.claimReview.some((r: any) =>
      (r.textualRating || '').toLowerCase() === 'true' ||
      (r.textualRating || '').toLowerCase() === 'false' ||
      (r.textualRating || '').toLowerCase() === 'misleading'
    )
  );

  if (verdict === 'REAL') {
    if (hasRelevantReview) {
      const supportingReview = gc.claims.find((c: any) =>
        c.claimReview.some((r: any) => (r.textualRating || '').toLowerCase() === 'true')
      );
      const publisher = supportingReview?.claimReview?.[0]?.publisher?.name || 'an independent fact-checker';
      return `Google's fact-check review (${publisher}) supports this claim, and TruthLensAI's own evidence also confirms it. Both sources agree on the verdict.`;
    }
    return 'Multiple independent, reliable sources directly support this claim with retrieved article text, meeting the evidence policy for a settled verdict. Google Fact Check did not return a directly relevant review for this claim.';
  }

  if (verdict === 'FAKE') {
    if (hasRelevantReview) {
      const contradictingReview = gc.claims.find((c: any) =>
        c.claimReview.some((r: any) => (r.textualRating || '').toLowerCase() === 'false')
      );
      const publisher = contradictingReview?.claimReview?.[0]?.publisher?.name || 'an independent fact-checker';
      return `Google's fact-check review (${publisher}) contradicts this claim, and TruthLensAI's own evidence also refutes it. Both sources agree on the verdict.`;
    }
    return 'Multiple independent publishers directly contradict this claim with retrieved article text, meeting the evidence policy for a settled verdict. Google Fact Check did not return a directly relevant review for this claim.';
  }

  if (verdict === 'UNVERIFIED') {
    if (hasRelevantReview) {
      return `Google's fact-check review addresses this claim, but TruthLensAI's own evidence is insufficient to reach a settled verdict. The review is shown for reference; the final verdict follows TruthLensAI's evidence requirements.`;
    }
    if (gc && gc.status === 'error') {
      return `Google Fact Check was unavailable for this request. TruthLensAI continued with its own evidence pipeline, but the available evidence is insufficient to confirm or refute the claim.`;
    }
    return 'No independent, reliable sources were found that directly establish or refute this claim. Google Fact Check did not return a matching review, and the available evidence does not meet the threshold for a confirmed verdict.';
  }

  if (verdict === 'MISLEADING') {
    return 'Independent sources confirm the core event but identify material distortions or omissions. Google\'s fact-check review, when available, is shown for comparison.';
  }

  return 'The available evidence does not establish this claim as true or false.';
}

export const UnifiedReport: React.FC<UnifiedReportProps> = ({
  result,
  modality,
  claimText,
  selectedFile,
}) => {
  const fact = result.factual_verification || {};
  const verdict = (fact.verdict || 'UNVERIFIED').toUpperCase();
  const confidence = fact.confidence;
  const confidenceBasis = fact.confidence_basis;
  const evidenceSufficiency = (fact.evidence_sufficiency || 'none').toLowerCase();
  const reasoning = fact.reasoning || fact.deterministic_gate_reason || '';
  const supportingEvidence = Array.isArray(fact.supporting_evidence) ? fact.supporting_evidence : [];
  const contradictingEvidence = Array.isArray(fact.contradicting_evidence) ? fact.contradicting_evidence : [];

  const v = VERDICT_STYLE[verdict] ?? VERDICT_STYLE.UNVERIFIED;

  // Short plain-English explanation, capped at 3 sentences.
  const shortExplanation = reasoning
    .split(/(?<=[.!?])\s+/)
    .filter((s) => s.trim().length > 0)
    .slice(0, 3)
    .join(' ')
    .trim();

  // Evidence status from the deterministic sufficiency label.
  const isSufficient =
    evidenceSufficiency.includes('strong_support') ||
    evidenceSufficiency.includes('strong_contradiction') ||
    (evidenceSufficiency.includes('support') && !evidenceSufficiency.includes('single')) ||
    (evidenceSufficiency.includes('contradiction') &&
      !evidenceSufficiency.includes('single') &&
      !evidenceSufficiency.includes('fallback'));
  const isLimited =
    evidenceSufficiency.includes('single') ||
    evidenceSufficiency.includes('fallback');
  const evidenceStatus = isSufficient
    ? 'SUFFICIENT'
    : isLimited
    ? 'LIMITED'
    : 'INSUFFICIENT';
  const es = EVIDENCE_STATUS[evidenceStatus] ?? EVIDENCE_STATUS.INSUFFICIENT;

  // Up to 2 most relevant verified evidence passages with publisher + URL.
  const keyEvidence = [...supportingEvidence, ...contradictingEvidence]
    .filter((e: any) => e.chunk && e.url)
    .slice(0, 2);

  // Show confidence only for settled verdicts with an evidence_strength basis.
  // verification_coverage (UNVERIFIED) is NOT a truth probability — omit it.
  const showConfidence =
    (verdict === 'REAL' || verdict === 'FAKE') &&
    confidence != null &&
    confidenceBasis === 'evidence_strength';

  const primaryClaim = result.primary_claim || claimText || '';

  // Google Fact Check data
  const gc = result.factual_verification?.google_fact_check ?? null;
  const hasGoogleResult = gc && gc.status === 'success' && gc.claims && gc.claims.length > 0;
  const hasRelevantReview = hasGoogleResult && gc.claims.some((c: any) =>
    c.claimReview && c.claimReview.some((r: any) =>
      (r.textualRating || '').toLowerCase() === 'true' ||
      (r.textualRating || '').toLowerCase() === 'false' ||
      (r.textualRating || '').toLowerCase() === 'misleading'
    )
  );

  // Independent LLM Analysis
  const independentLlm = fact.independent_llm_analysis || null;
  const hasIndependentLlm = !!independentLlm;

  // Image Origin Analysis — rendered only when the backend actually ran the
  // image-origin detector and returned probabilities. Kept strictly separate
  // from the claim-verification confidence; it says nothing about whether the
  // news claim itself is true.
  const synth = result.synthetic_media_detection || null;
  const imageOrigin =
    modality === 'image' && synth?.available
      ? {
          signal: (synth.signal || 'UNAVAILABLE').toUpperCase(),
          ai: synth.probabilities?.ai_probability,
          real: synth.probabilities?.real_probability,
        }
      : null;
  const hasImageOrigin =
    !!imageOrigin && (imageOrigin.ai != null || imageOrigin.real != null);

  const formatProbability = (value: number | null | undefined): string | null => {
    if (value === null || value === undefined || Number.isNaN(value)) {
      return null;
    }
    return (value > 1 ? value : value * 100).toFixed(1);
  };

  const aiPercent = formatProbability(imageOrigin?.ai ?? null);
  const realPercent = formatProbability(imageOrigin?.real ?? null);

  return (
    <div className="bg-surface-container rounded-xl border border-outline-variant/50 shadow-md p-6 sm:p-8 text-on-surface">
      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <span className="material-symbols-outlined text-primary text-[24px]">verified</span>
        <h2 className="text-sm font-mono uppercase tracking-wider font-bold text-tertiary">
          TRUTHLENS AI — FINAL ASSESSMENT
        </h2>
      </div>

      {/* Error state */}
      {result.status === 'error' && (
        <div className="p-4 rounded-lg bg-error-container/20 border border-error/40 mb-5">
          <span className="material-symbols-outlined text-error text-[18px] flex-shrink-0 mr-2">
            error
          </span>
          <div className="flex flex-col gap-1">
            <span className="text-[10px] font-mono uppercase text-error font-semibold block">
              Verification Failed
            </span>
            <p className="text-sm text-on-surface-variant leading-relaxed">
              {result.error || 'The verification request could not be completed.'}
            </p>
          </div>
        </div>
      )}

      {/* Claim */}
      <div className="mb-5">
        <span className="text-[10px] font-mono uppercase tracking-wider text-outline block mb-1">
          Claim
        </span>
        <p className="text-sm font-sans text-on-surface font-medium leading-relaxed bg-surface-container-low p-3 rounded-lg border border-outline-variant/30">
          "{primaryClaim}"
        </p>
        {claimText !== primaryClaim && (
          <div className="flex flex-col gap-1 mt-2">
            <span className="text-[10px] font-mono uppercase text-outline">
              Original Text
            </span>
            <p className="text-sm font-sans text-on-surface leading-relaxed bg-surface-container-low p-3 rounded-lg border border-outline-variant/30">
              "{claimText}"
            </p>
          </div>
        )}
      </div>

      {/* Verdict — strongest visual element */}
      <div className="flex flex-col items-center justify-center gap-3 mb-5 p-5 rounded-xl border-2 border-outline-variant/40 bg-surface-container-low">
        <span className="text-[10px] font-mono uppercase tracking-wider text-outline block">
          Verdict
        </span>
        <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full border font-mono text-lg font-bold uppercase tracking-wider ${v.badge}`}>
          <span className={`w-2.5 h-2.5 rounded-full ${v.dot}`} />
          {v.label}
        </span>
        <p className="text-xs font-mono text-on-surface-variant max-w-md text-center leading-relaxed">
          {shortExplanation}
        </p>
      </div>

      {/* Final Verdict Explanation */}
      <div className="mb-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="material-symbols-outlined text-[16px] text-secondary">description</span>
          <span className="text-[10px] font-mono uppercase tracking-wider text-outline font-semibold">
            Final Verdict Explanation
          </span>
        </div>
        <div className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/30">
          <p className="text-sm font-sans text-on-surface leading-relaxed">
            {generateFinalVerdictExplanation(
              verdict,
              fact.independent_supporting || 0,
              fact.independent_contradicting || 0,
              evidenceSufficiency,
              reasoning,
              primaryClaim
            )}
          </p>
        </div>
      </div>

      {/* AI Confidence */}
      <div className="mb-5">
        <div className="flex items-center gap-2 mb-1">
          <span className="material-symbols-outlined text-[16px] text-secondary">analytics</span>
          <span className="text-[10px] font-mono uppercase tracking-wider text-outline font-semibold">
            AI Confidence
          </span>
        </div>
        {showConfidence ? (
          <span className="text-sm font-mono font-bold text-on-surface">
            {(confidence > 1 ? confidence : confidence * 100).toFixed(1)}%
          </span>
        ) : (
          <span className="text-sm font-mono text-on-surface-variant">Confidence unavailable.</span>
        )}
        <p className="text-[10px] text-outline/70 mt-1 leading-relaxed">
          Final claim-verification confidence from the backend evidence assessment.
        </p>
      </div>

      {/* Image Origin Analysis — rendered only when the backend actually ran the
      image-origin detector and returned probabilities. Kept strictly separate
      from the claim-verification confidence; it says nothing about whether the
      news claim itself is true. */}
      {hasImageOrigin && (
        <div className="mb-5">
          <div className="flex items-center gap-2 mb-2">
            <span className="material-symbols-outlined text-[16px] text-secondary">image</span>
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline font-semibold">
              Image Origin Analysis
            </span>
          </div>
          <ul className="space-y-1.5 text-sm font-mono text-on-surface">
            <li className="flex justify-between">
              <span>AI-generated probability</span>
              <span className="font-bold">{aiPercent ?? 'n/a'}%</span>
            </li>
            <li className="flex justify-between">
              <span>Real-image probability</span>
              <span className="font-bold">{realPercent ?? 'n/a'}%</span>
            </li>
            <li className="flex justify-between">
              <span>Origin assessment</span>
              <span className="font-bold uppercase tracking-wide">
                {imageOrigin.signal}
              </span>
            </li>
          </ul>
          <p className="text-[10px] text-outline/70 mt-2 leading-relaxed">
            Independent signal about whether the image itself is AI-generated or
            a real photograph. It is not a probability that the news claim is
            true.
          </p>
        </div>
      )}

      {/* Independent LLM Analysis */}
      {hasIndependentLlm ? (
        <div className="mb-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="material-symbols-outlined text-[16px] text-secondary">psychology</span>
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline font-semibold">
              Independent LLM Analysis
            </span>
          </div>
          <div className="mb-2">
            <span className="text-[9px] font-mono uppercase text-outline">
              Claim Summary
            </span>
            <p className="text-sm text-on-surface font-sans leading-relaxed bg-surface-container-low p-3 rounded-lg border border-outline-variant/30">
              {independentLlm.claim_summary}
            </p>
          </div>
          <div className="mb-2">
            <span className="text-[9px] font-mono uppercase text-outline">
              Independent LLM Verdict
            </span>
            <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border font-mono text-xs font-bold uppercase tracking-wider ${VERDICT_STYLE[independentLlm.verdict ?? 'UNVERIFIED']?.badge ?? VERDICT_STYLE.UNVERIFIED.badge}`}>
              <span className={`w-2 h-2 rounded-full ${VERDICT_STYLE[independentLlm.verdict ?? 'UNVERIFIED']?.dot ?? VERDICT_STYLE.UNVERIFIED.dot}`} />
              {independentLlm.verdict ?? '—'}
            </span>
            {independentLlm.confidence != null && (
              <span className="text-[10px] text-outline/70 ml-2">
                ({independentLlm.confidence.toFixed(1)}%)
              </span>
            )}
          </div>
          {independentLlm.reasoning && (
            <div className="mb-2">
              <span className="text-[9px] font-mono uppercase text-outline block mb-1">
                Independent LLM Reasoning
              </span>
              <p className="text-sm font-sans text-on-surface leading-relaxed bg-surface-container-low p-3 rounded-lg border border-outline-variant/30">
                {independentLlm.reasoning}
              </p>
            </div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2">
            {(independentLlm.supporting_evidence.length > 0 ||
              independentLlm.contradicting_evidence.length > 0) && (
              <div className="flex flex-col gap-1.5">
                <span className="text-[9px] font-mono uppercase text-outline block mb-1">
                  Evidence References
                </span>
                {independentLlm.supporting_evidence.map((ref, i) => (
                  <div key={i} className="flex items-start gap-1.5 text-[11px] text-on-surface-variant leading-relaxed">
                    <span className="text-emerald-400 flex-shrink-0">+</span>
                    <span>
                      {ref.excerpt}
                      {' '}
                      <span className="text-outline/60 font-mono">({ref.domain})</span>
                    </span>
                  </div>
                ))}
                {independentLlm.contradicting_evidence.map((ref, i) => (
                  <div key={i} className="flex items-start gap-1.5 text-[11px] text-on-surface-variant leading-relaxed">
                    <span className="text-rose-400 flex-shrink-0">-</span>
                    <span>
                      {ref.excerpt}
                      {' '}
                      <span className="text-outline/60 font-mono">({ref.domain})</span>
                    </span>
                  </div>
                ))}
              </div>
            )}
            {(independentLlm.limitations.length > 0 || independentLlm.uncertainty) && (
              <div className="mt-2 p-2.5 rounded-lg bg-surface-container-low border border-outline-variant/30 flex flex-col gap-1">
                {independentLlm.uncertainty && (
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[9px] font-mono uppercase text-outline block mb-1">
                      Uncertainty
                    </span>
                    <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                      {independentLlm.uncertainty}
                    </p>
                  </div>
                )}
                {independentLlm.limitations.length > 0 && (
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[9px] font-mono uppercase text-outline block">
                      Limitations
                    </span>
                    <ul className="flex flex-col gap-0.5 text-[11px] text-on-surface-variant font-sans list-disc list-inside">
                      {independentLlm.limitations.slice(0, 3).map((lim, i) => (
                        <li key={i}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <p className="text-[9px] text-outline/70 mt-1 leading-relaxed">
                  {independentLlm.comparison?.comparison_summary ?? ''}
                </p>
              </div>
            )}
          </div>
        </div>
      ) : null}

      {/* Google Fact Check */}
      {hasGoogleResult && (
        <div className="mb-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="material-symbols-outlined text-[16px] text-secondary">fact_check</span>
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline font-semibold">
              Google Fact Check Result
            </span>
          </div>
          {gc.claims.map((claimResult: any, i: number) => (
            <div key={i} className="p-3.5 rounded-lg bg-surface-container-low border border-outline-variant/30">
              <p className="text-sm text-on-surface font-sans leading-relaxed mb-2">
                "{claimResult.text}"
              </p>
              {claimResult.claimReview.map((review: any, j: number) => (
                <div key={j} className="p-3 rounded bg-surface-container-low border border-outline-variant/30">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[9px] font-mono uppercase text-outline">Publisher</span>
                    <span className="text-xs font-sans text-on-surface">{review.publisher?.name || 'Unknown'}</span>
                  </div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-[9px] font-mono uppercase text-outline">Rating</span>
                    <span className={`text-xs font-mono font-bold ${review.textualRating?.toLowerCase() === 'true' ? 'text-emerald-400' : review.textualRating?.toLowerCase() === 'false' ? 'text-rose-400' : 'text-outline'}`}>
                      {review.textualRating}
                    </span>
                  </div>
                  {review.url && (
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[9px] font-mono uppercase text-outline">Source</span>
                      <a href={review.url} target="_blank" rel="noopener noreferrer" className="text-[10px] font-mono text-primary hover:underline underline-offset-1 truncate max-w-[200px]">
                        {review.url}
                      </a>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

    </div>
  );
};
