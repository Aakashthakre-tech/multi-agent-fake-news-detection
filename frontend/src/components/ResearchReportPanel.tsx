import React, { useState } from 'react';
import {
  ResearchFinding,
  ResearchReport as ResearchReportModel,
  UnifiedVerificationResponse,
} from '../types';
import ClaimBreakdownCard from './ClaimBreakdownCard';

/**
 * ResearchReportPanel.tsx
 * ========================
 * Renders the point-wise, evidence-grounded research report produced by the
 * backend (`research_report.py`). Every value shown here comes from the API
 * response — the panel never recomputes a verdict, invents a source, or
 * fabricates a score.
 */

interface Props {
  result: UnifiedVerificationResponse;
}

const isSafeUrl = (url?: string | null): boolean => {
  if (!url) return false;
  const candidate = url.trim();
  return (
    (candidate.startsWith('http://') || candidate.startsWith('https://')) &&
    !/\s/.test(candidate)
  );
};

const pct = (v?: number | null): string => {
  if (v == null || Number.isNaN(v)) return 'N/A';
  return v > 1 ? `${v.toFixed(1)}%` : `${(v * 100).toFixed(1)}%`;
};

const STATUS_STYLE: Record<string, { label: string; cls: string }> = {
  SUPPORTED: { label: 'SUPPORTED', cls: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' },
  CONTRADICTED: { label: 'CONTRADICTED', cls: 'text-rose-400 border-rose-500/30 bg-rose-500/10' },
  CONTEXT: { label: 'CONTEXT', cls: 'text-sky-400 border-sky-500/30 bg-sky-500/10' },
  NON_EVIDENCE: { label: 'NOT EVIDENCE', cls: 'text-outline border-outline-variant/40 bg-surface-container-low' },
};

const FindingCard: React.FC<{ finding: ResearchFinding; index: number }> = ({ finding, index }) => {
  const [open, setOpen] = useState(false);
  const style = STATUS_STYLE[finding.status] || STATUS_STYLE.NON_EVIDENCE;
  const linkable = isSafeUrl(finding.source_url);

  return (
    <div className="rounded-lg border border-outline-variant/30 bg-surface-container-low overflow-hidden">
      <div
        onClick={() => setOpen((v) => !v)}
        className="flex items-start justify-between gap-3 p-3 cursor-pointer"
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[10px] font-mono text-tertiary">#{index + 1}</span>
            <span className={`text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border ${style.cls}`}>
              {style.label}
            </span>
            {finding.evidence_strength != null && (
              <span className="text-[9px] font-mono text-outline">
                evidence strength {finding.strength_label} ({pct(finding.evidence_strength)})
              </span>
            )}
          </div>
          <p className="text-xs font-semibold text-on-surface font-sans mt-1 leading-snug">
            {finding.title}
          </p>
          {finding.source_name && (
            <p className="text-[10px] font-mono text-outline mt-0.5">
              {finding.source_name}
              {finding.source_domain && finding.source_domain !== finding.source_name
                ? ` — ${finding.source_domain}`
                : ''}
              {finding.source_tier ? ` | tier: ${finding.source_tier}` : ''}
              {finding.source_reliability != null
                ? ` | reliability ${finding.source_reliability.toFixed(2)}`
                : ''}
            </p>
          )}
        </div>
        <span className="material-symbols-outlined text-[16px] text-outline flex-shrink-0">
          {open ? 'expand_less' : 'expand_more'}
        </span>
      </div>

      {open && (
        <div className="px-3 pb-3 flex flex-col gap-2 border-t border-outline-variant/20 pt-2">
          <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
            {finding.explanation}
          </p>

          {finding.claim_text && (
            <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
              <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                Related claim
              </span>
              <p className="text-[11px] text-on-surface font-sans">{finding.claim_text}</p>
            </div>
          )}

          {finding.excerpt && (
            <div className="p-2 rounded bg-surface-container-lowest border border-outline-variant/20">
              <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                Retrieved passage{finding.excerpt_kind === 'verbatim_passage' ? ' (verbatim)' : ''}
              </span>
              <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                {finding.excerpt}
              </p>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[9px] font-mono text-outline">
            {finding.nli_label && <span>NLI {finding.nli_label}</span>}
            {finding.nli_score != null && <span>score {finding.nli_score.toFixed(3)}</span>}
            {finding.date_status && <span>date: {finding.date_status}</span>}
            {finding.event_match && <span>event: {finding.event_match}</span>}
            {finding.evidence_type && <span>type: {finding.evidence_type}</span>}
            {finding.independence_group && <span>group: {finding.independence_group}</span>}
            {finding.search_query && <span>query: “{finding.search_query}”</span>}
          </div>

          {linkable && (
            <a
              href={finding.source_url as string}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1 text-[10px] font-mono text-primary hover:underline"
            >
              <span className="material-symbols-outlined text-[12px]">open_in_new</span>
              Visit source
            </a>
          )}
        </div>
      )}
    </div>
  );
};

const FindingGroup: React.FC<{
  title: string;
  icon: string;
  tone: string;
  items: ResearchFinding[];
  emptyNote?: string;
}> = ({ title, icon, tone, items, emptyNote }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="rounded-lg border border-outline-variant/30 bg-surface-container-lowest">
      <div
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between gap-2 p-2.5 cursor-pointer"
      >
        <div className="flex items-center gap-2">
          <span className={`material-symbols-outlined text-[15px] ${tone}`}>{icon}</span>
          <span className="text-[10px] font-mono uppercase tracking-widest text-on-surface">
            {title}
          </span>
          <span className="text-[10px] font-mono text-outline">({items.length})</span>
        </div>
        <span className="material-symbols-outlined text-[15px] text-outline">
          {open ? 'expand_less' : 'expand_more'}
        </span>
      </div>
      {open && (
        <div className="px-2.5 pb-2.5 flex flex-col gap-2">
          {items.length === 0 ? (
            <p className="text-[11px] font-sans text-outline italic py-1">
              {emptyNote || 'No findings in this category. The evidence did not produce any.'}
            </p>
          ) : (
            items.map((f, i) => <FindingCard key={`${f.source_url}-${i}`} finding={f} index={i} />)
          )}
        </div>
      )}
    </div>
  );
};

const ResearchReportPanel: React.FC<Props> = ({ result }) => {
  const report: ResearchReportModel | null = result.research_report || null;
  const [expanded, setExpanded] = useState(true);

  if (!report) {
    return (
      <div className="bg-surface-container rounded-xl p-4 border border-outline-variant/40">
        <p className="text-xs font-sans text-outline">
          No research report was returned for this result.
        </p>
      </div>
    );
  }

  const summary = report.executive_summary || ({} as ResearchReportModel['executive_summary']);
  const findings = report.findings || {};
  const method = report.methodology || ({} as ResearchReportModel['methodology']);
  const claims = report.claims || [];
  const gaps = report.evidence_gaps || [];

  const supporting = findings.supporting || [];
  const contradicting = findings.contradicting || [];
  const contextual = findings.contextual || [];
  const unresolved = findings.unresolved || [];

  return (
    <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/50 shadow-md flex flex-col gap-4">
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                            */}
      {/* ---------------------------------------------------------------- */}
      <div
        onClick={() => setExpanded((v) => !v)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-primary">travel_explore</span>
          <span className="text-xs font-mono uppercase tracking-widest text-primary font-semibold">
            Point-Wise Research Report
          </span>
        </div>
        <span className="material-symbols-outlined text-[16px] text-outline">
          {expanded ? 'expand_less' : 'expand_more'}
        </span>
      </div>

      {expanded && (
        <div className="flex flex-col gap-4 pt-3 border-t border-outline-variant/30">
          {/* ------------------------------------------------------------ */}
          {/* 1. Executive summary                                            */}
          {/* ------------------------------------------------------------ */}
          <section className="flex flex-col gap-2">
            <h4 className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
              1. Executive Summary
            </h4>

            {summary.input_claim && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  Input under investigation
                </span>
                <p className="text-xs text-on-surface font-sans leading-relaxed">
                  {summary.input_claim}
                </p>
              </div>
            )}

            {/* --- Deterministic-gate policy accounting ------------------- */}
            {summary.policy_diagnostics && (
              <div
                className={
                  'p-2.5 rounded border ' +
                  (summary.policy_diagnostics.verdict_restricted
                    ? 'bg-surface-container-lowest border-amber-500/40'
                    : 'bg-surface-container-lowest border-outline-variant/20')
                }
              >
                <div className="flex items-center gap-2 flex-wrap mb-1">
                  <span className="material-symbols-outlined text-[14px] text-outline">
                    {summary.policy_diagnostics.verdict_restricted
                      ? 'report_problem'
                      : 'verified'}
                  </span>
                  <span className="text-[9px] font-mono uppercase text-outline">
                    Evidence policy accounting
                  </span>
                  <span
                    className={
                      'text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border ' +
                      (summary.policy_diagnostics.verdict_restricted
                        ? 'text-amber-400 border-amber-500/40 bg-amber-500/10'
                        : 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10')
                    }
                  >
                    {summary.policy_diagnostics.verdict_restricted
                      ? 'POLICY DATA INVALID'
                      : summary.policy_diagnostics.gate_ran
                        ? 'POLICY DATA VALID'
                        : 'GATE NOT RUN'}
                  </span>
                </div>
                <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                  {summary.policy_diagnostics.explanation}
                </p>
                {summary.policy_diagnostics.policy_data_errors?.length > 0 && (
                  <ul className="flex flex-col gap-0.5 mt-1.5">
                    {summary.policy_diagnostics.policy_data_errors.map((e, i) => (
                      <li key={i} className="text-[10px] font-mono text-amber-400/90">
                        • {e}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {summary.attempted && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  What the system attempted to verify
                </span>
                <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                  {summary.attempted}
                </p>
              </div>
            )}

            {summary.key_findings?.length > 0 && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-1">
                  Most important findings
                </span>
                <ul className="flex flex-col gap-1">
                  {summary.key_findings.map((line, i) => (
                    <li key={i} className="text-[11px] text-on-surface font-sans leading-relaxed flex gap-2">
                      <span className="text-primary font-mono">•</span>
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {(summary.why_this_verdict || summary.deterministic_gate_reason) && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  Why the verdict was reached
                </span>
                <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                  {summary.why_this_verdict || summary.deterministic_gate_reason}
                </p>
                {summary.deterministic_gate_reason &&
                  summary.why_this_verdict &&
                  summary.deterministic_gate_reason !== summary.why_this_verdict && (
                    <p className="text-[10px] text-outline font-sans leading-relaxed mt-1.5">
                      Deterministic gate: {summary.deterministic_gate_reason}
                    </p>
                  )}
              </div>
            )}

            {summary.agent_narrative && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  Verification agent&rsquo;s analysis
                </span>
                <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed whitespace-pre-line">
                  {summary.agent_narrative}
                </p>
              </div>
            )}

            {summary.critic_review && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  Critic agent&rsquo;s review
                </span>
                <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed whitespace-pre-line">
                  {summary.critic_review}
                </p>
              </div>
            )}

            {summary.limitations?.length > 0 && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-amber-500/25">
                <span className="text-[9px] font-mono uppercase text-amber-400/80 block mb-1">
                  Uncertainties &amp; limitations
                </span>
                <ul className="flex flex-col gap-1">
                  {summary.limitations.map((line, i) => (
                    <li key={i} className="text-[11px] text-on-surface-variant font-sans leading-relaxed flex gap-2">
                      <span className="text-amber-400/80 font-mono">!</span>
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {summary.confidence != null && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/20">
                <span className="text-[9px] font-mono uppercase text-outline block mb-0.5">
                  {summary.confidence_basis === 'verification_coverage'
                    ? 'Evidence coverage'
                    : 'Confidence'}
                  {summary.confidence_basis ? ` (${summary.confidence_basis})` : ''}
                </span>
                <p className="text-[11px] text-on-surface font-mono">
                  {pct(summary.confidence)}
                </p>
                <p className="text-[10px] text-on-surface-variant font-sans leading-relaxed mt-1">
                  {summary.confidence_basis_note ||
                    summary.confidence_basis_explanation ||
                    'Derived from independent corroboration, source agreement and the fusion score.'}
                </p>
              </div>
            )}
          </section>

          {/* ------------------------------------------------------------ */}
          {/* 2. Point-wise findings                                         */}
          {/* ------------------------------------------------------------ */}
          <section className="flex flex-col gap-2">
            <h4 className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
              2. Point-wise Findings
            </h4>
            <FindingGroup
              title="Supporting findings"
              icon="verified"
              tone="text-emerald-400"
              items={supporting}
              emptyNote="No retrieved passage was classified as direct support for the claim."
            />
            <FindingGroup
              title="Contradicting findings"
              icon="gavel"
              tone="text-rose-400"
              items={contradicting}
              emptyNote="No retrieved passage was classified as a direct contradiction."
            />
            <FindingGroup
              title="Contextual findings"
              icon="article"
              tone="text-sky-400"
              items={contextual}
              emptyNote="No contextual passages were retrieved."
            />
            <FindingGroup
              title="Unresolved &amp; non-evidence passages"
              icon="help"
              tone="text-outline"
              items={unresolved}
              emptyNote="No unresolved passages were recorded."
            />
          </section>

          {/* ------------------------------------------------------------ */}
          {/* 3. Claim-by-claim verification                                  */}
          {/* ------------------------------------------------------------ */}
          {claims.length > 0 && (
            <section className="flex flex-col gap-2">
              <h4 className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
                3. Claim-by-Claim Verification
              </h4>
              {claims.map((claim, i) => (
                <ClaimBreakdownCard
                  key={i}
                  index={i}
                  claim={{
                    claim_text: claim.claim_text,
                    claim_verdict: claim.verdict,
                    confidence: claim.confidence,
                    evidence_sufficiency: claim.evidence_sufficiency,
                    reasoning: claim.reasoning,
                    selection_reason: claim.selection_reason,
                    source_modality: claim.source_modality,
                    priority_score: claim.priority_score,
                    timestamps: claim.timestamps as any,
                    frame_references: claim.frame_references,
                    supporting_evidence: claim.supporting_evidence,
                    contradicting_evidence: claim.contradicting_evidence,
                    limitations: claim.limitations,
                  }}
                />
              ))}
            </section>
          )}

          {/* ------------------------------------------------------------ */}
          {/* 4. Research methodology & agent activity                       */}
          {/* ------------------------------------------------------------ */}
          <section className="flex flex-col gap-2">
            <h4 className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
              4. Research Methodology &amp; Agent Activity
            </h4>

            {method.research_skipped && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-amber-500/25">
                <span className="text-[9px] font-mono uppercase text-amber-400/80 block mb-0.5">
                  Research was skipped
                </span>
                <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                  {method.research_skipped_reason ||
                    'No verifiable claim was available, so the evidence engine was not run.'}
                </p>
              </div>
            )}

            {method.stages?.length > 0 && (
              <div className="rounded-lg border border-outline-variant/30 bg-surface-container-lowest">
                {method.stages.map((stage, i) => (
                  <div
                    key={i}
                    className="flex items-start justify-between gap-3 px-2.5 py-2 border-b border-outline-variant/15 last:border-b-0"
                  >
                    <span className="text-[10px] font-mono text-on-surface shrink-0">
                      {stage.name}
                    </span>
                    <span className="text-[10px] font-mono text-outline text-right flex-1">
                      {stage.detail || stage.status}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {method.search_queries?.length > 0 && (
              <div className="rounded-lg border border-outline-variant/30 bg-surface-container-lowest">
                <div className="px-2.5 py-2 border-b border-outline-variant/15">
                  <span className="text-[9px] font-mono uppercase text-outline">
                    Search queries issued ({method.search_queries.length})
                  </span>
                </div>
                {method.search_queries.map((q, i) => (
                  <div
                    key={i}
                    className="px-2.5 py-2 border-b border-outline-variant/15 last:border-b-0 flex flex-col gap-0.5"
                  >
                    <span className="text-[10px] font-mono text-on-surface">“{q.query}”</span>
                    <span className="text-[9px] font-mono text-outline">
                      {q.label ? `plan: ${q.label}` : ''}
                      {q.intent ? ` | intent: ${q.intent}` : ''}
                      {q.topic ? ` | topic: ${q.topic}` : ''}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {method.scrapes_failed?.length > 0 && (
              <div className="rounded-lg border border-outline-variant/30 bg-surface-container-lowest">
                <div className="px-2.5 py-2 border-b border-outline-variant/15">
                  <span className="text-[9px] font-mono uppercase text-outline">
                    Failed retrieval attempts ({method.scrapes_failed.length})
                  </span>
                </div>
                {method.scrapes_failed.map((f, i) => (
                  <div
                    key={i}
                    className="px-2.5 py-2 border-b border-outline-variant/15 last:border-b-0 flex flex-col gap-0.5"
                  >
                    <span className="text-[10px] font-mono text-on-surface">
                      {f.title || f.domain || 'source'}
                    </span>
                    <span className="text-[9px] font-mono text-outline leading-relaxed">
                      {f.reason}
                      {f.fallback_used ? ` | fallback: ${f.fallback_used}` : ' | discarded (no usable fallback)'}
                    </span>
                    {isSafeUrl(f.url) && (
                      <a
                        href={f.url as string}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-[9px] font-mono text-primary hover:underline"
                      >
                        <span className="material-symbols-outlined text-[11px]">open_in_new</span>
                        Visit source
                      </a>
                    )}
                  </div>
                ))}
              </div>
            )}

            {method.search_errors?.length > 0 && (
              <div className="p-2.5 rounded bg-surface-container-lowest border border-amber-500/25">
                <span className="text-[9px] font-mono uppercase text-amber-400/80 block mb-0.5">
                  Search errors ({method.search_errors.length})
                </span>
                <ul className="flex flex-col gap-0.5">
                  {method.search_errors.map((e, i) => (
                    <li key={i} className="text-[10px] text-on-surface-variant font-mono leading-relaxed">
                      • {e}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          {/* ------------------------------------------------------------ */}
          {/* 5. Unresolved questions & evidence gaps                         */}
          {/* ------------------------------------------------------------ */}
          {gaps.length > 0 && (
            <section className="flex flex-col gap-2">
              <h4 className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
                5. Unresolved Questions &amp; Evidence Gaps
              </h4>
              {gaps.map((gap, i) => (
                <div
                  key={i}
                  className="p-2.5 rounded bg-surface-container-lowest border border-outline-variant/25"
                >
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded border border-outline-variant/40 text-outline">
                      {gap.status}
                    </span>
                    <span className="text-[11px] font-semibold text-on-surface font-sans">
                      {gap.question}
                    </span>
                  </div>
                  {gap.detail && (
                    <p className="text-[11px] text-on-surface-variant font-sans leading-relaxed">
                      {gap.detail}
                    </p>
                  )}
                  {gap.what_would_help && (
                    <p className="text-[10px] text-outline font-sans leading-relaxed mt-1">
                      What would help: {gap.what_would_help}
                    </p>
                  )}
                </div>
              ))}
            </section>
          )}
        </div>
      )}
    </div>
  );
};

export default ResearchReportPanel;
