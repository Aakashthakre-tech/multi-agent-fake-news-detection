import React, { useState } from 'react';
import { UnifiedVerificationResponse } from '../types';
import { confidenceLabel, confidenceNote, formatScore, scoreKindFor } from '../lib/scores';

interface FinalVerdictExplanationProps {
  result: UnifiedVerificationResponse;
}

export const FinalVerdictExplanation: React.FC<FinalVerdictExplanationProps> = ({ result }) => {
  const [showAgentConsensus, setShowAgentConsensus] = useState<boolean>(true);

  const fact = result.factual_verification || {};
  const raw = result.raw || {};
  const finalV = raw.final_verdict || {};

  const verdict = fact.verdict || 'UNVERIFIED';

  const getVerdictTheme = (v: string) => {
    switch (v) {
      case 'REAL':
      case 'SUPPORTED':
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
          badge: 'bg-emerald-500 text-surface-container-lowest',
          icon: 'verified',
          title: 'VERIFIED REAL // CORROBORATED',
          desc: 'Independently corroborated by two or more verified primary publishers with consistent temporal and event anchors.',
        };
      case 'FAKE':
      case 'CONTRADICTED':
        return {
          bg: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
          badge: 'bg-rose-500 text-surface-container-lowest',
          icon: 'cancel',
          title: 'VERIFIED FAKE // CONTRADICTED',
          desc: 'Directly contradicted or debunked by multiple independent primary reporting sources.',
        };
      case 'MISLEADING':
        return {
          bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
          badge: 'bg-amber-500 text-surface-container-lowest',
          icon: 'warning',
          title: 'MISLEADING // OUT OF CONTEXT',
          desc: 'The statement contains factual elements that have been distorted, misattributed, or taken out of their true context.',
        };
      case 'INCONCLUSIVE':
        return {
          bg: 'bg-slate-500/10 border-slate-500/30 text-slate-400',
          badge: 'bg-slate-600 text-slate-100',
          icon: 'question_mark',
          title: 'INCONCLUSIVE // INSUFFICIENT VERIFIABLE CLAIMS',
          desc: 'No verifiable factual claims could be extracted from this media. Acoustic or visual analysis completed independently.',
        };
      case 'NOT_EVALUATED':
        return {
          bg: 'bg-surface-container-high border-outline-variant/50 text-outline',
          badge: 'bg-surface-container-highest text-outline border border-outline-variant/40',
          icon: 'block',
          title: 'NOT EVALUATED // FACT-CHECK DISABLED',
          desc: 'Factual claim verification was not requested for this analysis. Synthetic media detection ran independently.',
        };
      default:
        return {
          bg: 'bg-surface-container-high border-outline-variant/50 text-tertiary',
          badge: 'bg-surface-container-highest text-secondary border border-secondary/40',
          icon: 'help',
          title: 'UNVERIFIED // INSUFFICIENT EVIDENCE',
          desc: 'Insufficient independent corroboration found across global news wires. Lack of corroboration does NOT prove the claim false.',
        };
    }
  };

  const theme = getVerdictTheme(verdict);
  const explanation = fact.deterministic_gate_reason || fact.reasoning || finalV.permitted_reason || 'Evidence layer completed analysis without narrative rationale.';

  // Identify reasons for uncertainty or limitations
  const limitations: string[] = [];
  if (verdict === 'UNVERIFIED') {
    if ((fact.independent_supporting ?? 0) === 1) {
      limitations.push('Single Source Threshold: Only 1 independent publisher corroborated this claim. TruthLensAI requires >= 2 independent primary publisher groups to eliminate single-source bias.');
    }
    if ((fact.independent_supporting ?? 0) === 0 && (fact.independent_contradicting ?? 0) === 0) {
      limitations.push('Wire Absence: No direct primary reporting or fact-checking dossiers specifically confirmed or refuted this proposition.');
    }
    if ((fact.independent_supporting ?? 0) > 0 && (fact.independent_contradicting ?? 0) > 0) {
      limitations.push('Publisher Conflict: Independent sources present conflicting accounts or unresolved temporal discrepancy.');
    }
    limitations.push('Epistemic Principle: A verdict of "UNVERIFIED" represents a lack of independent evidence, not affirmative proof of falsity.');
  }

  return (
    <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/50 shadow-md flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-outline-variant/30 pb-3">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-secondary">gavel</span>
          <span className="text-xs font-mono uppercase tracking-widest text-secondary font-semibold">
            Adjudicated Final Verdict & Reasoning
          </span>
        </div>
        <span className="text-[10px] font-mono text-outline">
          {fact.verdict_overridden_by_evidence
            ? 'EVIDENCE-LAYER VETO APPLIED'
            : 'EVIDENCE LAYER UPHELD'}
        </span>
      </div>

      {/* Primary Adjudication Banner */}
      <div className={`p-4 sm:p-5 rounded-xl border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 ${theme.bg}`}>
        <div className="flex items-center gap-3.5">
          <div className="w-12 h-12 rounded-xl bg-surface-container-lowest/80 flex items-center justify-center flex-shrink-0 shadow-sm border border-outline-variant/30">
            <span className="material-symbols-outlined text-[28px]">{theme.icon}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-[10px] font-mono uppercase tracking-wider text-outline">
              Final Adjudicated Disposition
            </span>
            <h2 className="font-serif text-2xl sm:text-3xl font-bold tracking-tight text-on-surface">
              {verdict}
            </h2>
            <p className="text-xs text-on-surface-variant font-sans mt-0.5 max-w-xl">
              {theme.desc}
            </p>
          </div>
        </div>

        {fact.confidence !== null && fact.confidence !== undefined && (
          <div className="flex flex-col items-start sm:items-end flex-shrink-0 pl-2 sm:border-l border-outline-variant/30">
            <span className="text-[10px] font-mono text-outline uppercase">
              {confidenceLabel(fact.confidence_basis)}
            </span>
            <div className="flex items-baseline gap-0.5">
              <span className="font-serif text-3xl font-bold text-primary">
                {(fact.confidence > 1 ? fact.confidence : fact.confidence * 100).toFixed(1)}
              </span>
              <span className="text-sm font-mono text-primary font-bold">%</span>
            </div>
            {confidenceNote(fact.confidence_basis) && (
              <span className="text-[9px] text-outline/80 font-sans leading-snug max-w-[13rem] text-right">
                {confidenceNote(fact.confidence_basis)}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Concise Evidence-Based Explanation */}
      <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/30 flex flex-col gap-2">
        <span className="text-[10px] font-mono uppercase tracking-wider text-outline flex items-center gap-1">
          <span className="material-symbols-outlined text-[14px] text-primary">format_quote</span>
          Authoritative Adjudication Explanation
        </span>
        <p className="text-xs sm:text-sm font-sans text-on-surface leading-relaxed">
          {explanation}
        </p>

        {/* Deterministic-gate policy accounting. Only surfaced when the
            evidence policy data was unusable, because that is the case in
            which the verdict was restricted for a reason unrelated to the
            evidence itself. */}
        {fact.policy_data_valid === false && (
          <div className="p-2.5 rounded border border-amber-500/40 bg-amber-500/5 flex flex-col gap-1">
            <span className="text-[9px] font-mono uppercase text-amber-400 flex items-center gap-1">
              <span className="material-symbols-outlined text-[13px]">report_problem</span>
              Evidence policy data invalid
            </span>
            <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
              The deterministic gate could not evaluate the evidence policy, so the
              result was restricted to UNVERIFIED. This is an internal
              evidence-accounting issue, NOT proof that the claim is false.
            </p>
            {(fact.policy_data_errors?.length ?? 0) > 0 && (
              <ul className="flex flex-col gap-0.5">
                {fact.policy_data_errors!.map((e, i) => (
                  <li key={i} className="text-[10px] font-mono text-amber-400/90">
                    • {e}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Source Corroboration Summary */}
        <div className="flex flex-wrap items-center gap-3 pt-2 mt-1 border-t border-outline-variant/20 text-[10px] font-mono text-outline">
          <span>
            Supporting Sources: <strong className="text-emerald-400 font-bold">{fact.independent_supporting ?? 0} independent</strong>
          </span>
          <span>•</span>
          <span>
            Contradicting Sources: <strong className="text-rose-400 font-bold">{fact.independent_contradicting ?? 0} independent</strong>
          </span>
          <span>•</span>
          <span>
            Total Pool: <strong className="text-on-surface">{fact.total_sources || finalV.total_sources || 0} examined</strong>
          </span>
          <span>•</span>
          <span>
            Evidence Sufficiency: <strong className="text-on-surface">{fact.evidence_sufficiency || 'N/A'}</strong>
          </span>
        </div>

        {/* Evidence scores behind the verdict */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
          {[
            { label: 'Source Agreement', field: 'source_agreement' },
            { label: 'Fusion Score', field: 'fusion_score' },
            { label: 'Entailment Score', field: 'entailment_score' },
            { label: 'Contradiction Score', field: 'contradiction_score' },
          ].map((metric) => (
            <div
              key={metric.field}
              className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20 flex flex-col gap-0.5"
            >
              <span className="text-[9px] text-outline uppercase">{metric.label}</span>
              <span className="text-on-surface font-bold">
                {formatScore(
                  (fact as unknown as Record<string, number | null | undefined>)[metric.field],
                  scoreKindFor(metric.field)
                )}
              </span>
            </div>
          ))}
        </div>

        {/* The verdicts the deterministic evidence floor permitted */}
        {fact.permitted_verdicts && fact.permitted_verdicts.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono text-outline">
            <span>Verdicts permitted by the evidence floor:</span>
            {fact.permitted_verdicts.map((v) => (
              <span
                key={v}
                className="px-1.5 py-0.5 rounded border border-outline-variant/30 text-on-surface"
              >
                {v}
              </span>
            ))}
          </div>
        )}

        {/* A partial verification must not read as a finished one. */}
        {result.status && result.status !== 'success' && (
          <div className="p-2.5 rounded border border-amber-500/40 bg-amber-500/5 flex flex-col gap-1">
            <span className="text-[9px] font-mono uppercase text-amber-400 flex items-center gap-1">
              <span className="material-symbols-outlined text-[13px]">pending</span>
              Partial verification — processing status: {result.status.toUpperCase()}
            </span>
            <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
              The evidence was retrieved and adjudicated, but not every stage
              completed. The verdict below reflects the evidence that was
              actually collected.
            </p>
          </div>
        )}
      </div>

      {/* Uncertainty & Limitations Panel (Crucial for Epistemic Integrity) */}
      {limitations.length > 0 && (
        <div className="bg-surface-container-lowest p-3.5 rounded-lg border border-amber-500/20 flex flex-col gap-1.5">
          <span className="text-[10px] font-mono uppercase tracking-wider text-amber-400 font-semibold flex items-center gap-1">
            <span className="material-symbols-outlined text-[14px]">info</span>
            Limitations, Uncertainty & Epistemic Boundaries
          </span>
          <ul className="flex flex-col gap-1 text-[11px] text-on-surface-variant font-sans list-disc list-inside">
            {limitations.map((lim, i) => (
              <li key={i} className="leading-relaxed">
                {lim}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Multi-Agent Consensus & Deterministic Gate Audit */}
      {(finalV.verification_verdict || fact.agent_verdict) && (
        <div className="bg-surface-container-low p-3.5 rounded-lg border border-outline-variant/30 flex flex-col gap-2.5">
          <div
            onClick={() => setShowAgentConsensus(!showAgentConsensus)}
            className="flex items-center justify-between cursor-pointer"
          >
            <span className="text-[10px] font-mono uppercase tracking-wider text-secondary font-semibold flex items-center gap-1">
              <span className="material-symbols-outlined text-[14px]">smart_toy</span>
              Multi-Agent Audit (LLaMA Agent vs Deterministic Gate)
            </span>
            <span className="material-symbols-outlined text-[16px] text-outline">
              {showAgentConsensus ? 'expand_less' : 'expand_more'}
            </span>
          </div>

          {showAgentConsensus && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] font-mono pt-1">
              {/* Primary Agent */}
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20 flex flex-col gap-0.5">
                <span className="text-[9px] text-outline uppercase">Verification Agent:</span>
                <span className="text-tertiary font-bold">
                  {finalV.verification_verdict || fact.agent_verdict || 'N/A'}
                </span>
                <span className="text-[10px] text-outline">
                  Agent-stated confidence:{' '}
                  {finalV.agent_confidence ?? fact.agent_confidence ?? 'N/A'}%
                </span>
                <span className="text-[9px] text-outline/80 font-sans leading-snug">
                  How sure the agent was in its own reading. It is not a
                  probability that the claim is true.
                </span>
              </div>

              {/* Critic Agent */}
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20 flex flex-col gap-0.5">
                <span className="text-[9px] text-outline uppercase">Adversarial Critic:</span>
                <span className="text-tertiary font-bold">
                  {finalV.critic_verdict || fact.critic_verdict || 'N/A'}
                </span>
                <span className="text-[10px] text-outline">
                  Agent-stated confidence:{' '}
                  {finalV.critic_confidence ?? fact.critic_confidence ?? 'N/A'}%
                </span>
                <span className="text-[9px] text-outline/80 font-sans leading-snug">
                  How sure the critic was in its own review. It is not a
                  probability that the claim is true.
                </span>
              </div>

              {/* Deterministic Evidence Gate Status */}
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20 flex flex-col gap-0.5">
                <span className="text-[9px] text-outline uppercase">Evidence Layer Action:</span>
                <span className={`font-bold ${finalV.verdict_overridden_by_evidence ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {finalV.verdict_overridden_by_evidence ? 'OVERRODE AGENT (VETO)' : 'UPHELD AGENT'}
                </span>
                <span className="text-[10px] text-outline">
                  {finalV.verdict_overridden_by_evidence
                    ? 'Evidence gate forced UNVERIFIED'
                    : 'Evidence threshold met'}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
