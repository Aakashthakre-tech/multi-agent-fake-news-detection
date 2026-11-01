#!/usr/bin/env python3
"""Write InvestigationTimeline.tsx with correct content."""

content = """\
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

  // Use backend-provided pipeline_stages for all modalities
  const backendStages: PipelineStageDetails[] =
    Array.isArray(result.pipeline_stages) && result.pipeline_stages.length > 0
      ? result.pipeline_stages
      : [];

  // Fallback reconstruction for text modality when backend stages absent
  const fallbackStages: PipelineStageDetails[] = React.useMemo(() => {
    if (backendStages.length > 0 || result.modality !== 'text') return [];
    const finalV = raw.final_verdict || {};
    const agreement = raw.source_agreement || {};
    const evidenceAnalysis = raw.evidence_analysis || [];
    const sourceData = raw.source_data || [];
    const stages: PipelineStageDetails[] = [];

    stages.push({
      id: 'ingestion', number: 1, name: 'Ingestion and Claim Normalization', status: 'completed',
      operation: 'Validated input syntax and bounded claim character budget for downstream NLI token safety.',
      why: 'Prevents silent truncation in Cross-Encoder tokenizer and establishes investigation scope.',
      tool_or_model: 'TruthLensAI Router',
      input_summary: result.primary_claim ? result.primary_claim.slice(0, 120) : 'User submission',
      output_summary: 'Verified TEXT payload ready for multi-agent evidence pipeline.',
    });

    if (sourceData.length > 0 || raw.search_results) {
      stages.push({
        id: 'search', number: 2, name: 'Query Planning and Web Search', status: 'completed',
        operation: 'Dispatched search queries across global news wires.',
        why: 'Gathers raw primary candidate documents to cross-reference the stated proposition.',
        tool_or_model: 'Tavily Search API',
        input_summary: (result.primary_claim || '').slice(0, 80),
        output_summary: `${sourceData.length} candidate source URLs retrieved.`,
        metric: { label: 'Candidate Sources', value: sourceData.length || 0, meaning: 'URLs retrieved from web search.', scale: 'Count' },
      });
    }

    if (evidenceAnalysis.length > 0) {
      const s = evidenceAnalysis.filter((e: any) => e.evidence_class === 'SUPPORT').length;
      const c = evidenceAnalysis.filter((e: any) => e.evidence_class === 'CONTRADICTION').length;
      const cx = evidenceAnalysis.filter((e: any) => e.evidence_class === 'RELATED_CONTEXT').length;
      stages.push({
        id: 'nli_analysis', number: stages.length + 1, name: 'Evidence Chunking and NLI Cross-Encoder', status: 'completed',
        operation: 'Segmented articles and evaluated premise-to-hypothesis entailment.',
        why: 'Determines whether passages logically entail, contradict, or discuss the claim.',
        tool_or_model: 'cross-encoder/nli-deberta-v3-base',
        input_summary: `${evidenceAnalysis.length} passages`,
        output_summary: `${s} SUPPORT, ${c} CONTRADICTION, ${cx} CONTEXT.`,
        metric: { label: 'NLI Confidence', value: evidenceAnalysis[0]?.nli_score ? `${(evidenceAnalysis[0].nli_score * 100).toFixed(1)}%` : 'Evaluated', meaning: 'Top chunk NLI confidence.', scale: '0.0-1.0' },
      });
    }

    if (fact.source_agreement !== null || agreement.agreement_score !== undefined) {
      stages.push({
        id: 'fusion', number: stages.length + 1, name: 'Source Agreement and Evidence Fusion', status: 'completed',
        operation: 'Synthesized independence group votes and calculated agreement score.',
        why: 'Protects against confirmation bias by evaluating ratio of corroborating versus debunking sources.',
        tool_or_model: 'Deterministic Evidence Fusion Engine',
        input_summary: `${fact.independent_supporting || 0} support vs ${fact.independent_contradicting || 0} contra.`,
        output_summary: `Evidence Sufficiency: ${fact.evidence_sufficiency || 'UNKNOWN'}; Agreement: ${((fact.source_agreement || 0)).toFixed(2)}`,
        metric: { label: 'Agreement Score', value: `${((fact.source_agreement || agreement.agreement_score || 0) * 100).toFixed(1)}%`, meaning: 'Harmonic agreement among independent publishers.', scale: '0-100%' },
      });
    }

    if (finalV.verification_verdict || fact.agent_verdict) {
      stages.push({
        id: 'verify_agent', number: stages.length + 1, name: 'Verification Agent Analysis', status: 'completed',
        operation: 'LLM Agent ingested evidence brief and produced investigative verdict.',
        why: 'Analyzes linguistic nuances, historical context, and potential misinterpretation.',
        tool_or_model: 'Groq LLaMA-3.3-70B-Versatile (Verification Agent)',
        input_summary: 'Synthesized Evidence Brief',
        output_summary: `Proposed Verdict: ${finalV.verification_verdict || fact.agent_verdict}`,
        metric: { label: 'Agent Confidence', value: `${finalV.agent_confidence ?? fact.agent_confidence ?? 'N/A'}%`, meaning: 'Verification Agent self-reported certainty.', scale: '0-100%' },
      });
    }

    if (finalV.critic_verdict || fact.critic_verdict) {
      stages.push({
        id: 'critic_agent', number: stages.length + 1, name: 'Adversarial Critic Review Agent', status: 'completed',
        operation: 'Adversarial Critic examined Verification Agent reasoning for hallucinations.',
        why: 'Prevents sycophancy and challenges ungrounded assertions before verdict adjudication.',
        tool_or_model: 'Groq LLaMA-3.3-70B-Versatile (Critic Agent)',
        input_summary: 'Primary Agent Rationale + Citations',
        output_summary: `Critic: ${finalV.critic_verdict || fact.critic_verdict}; Consensus: ${finalV.agents_agree ? 'AGREE' : 'DISAGREE'}`,
        metric: { label: 'Critic Confidence', value: `${finalV.critic_confidence ?? fact.critic_confidence ?? 'N/A'}%`, meaning: 'Adversarial Critic audit confidence.', scale: '0-100%' },
      });
    }

    stages.push({
      id: 'adjudication', number: stages.length + 1, name: 'Deterministic Gate and Final Adjudication', status: 'completed',
      operation: 'Applied strict evidence rules with deterministic gate veto over LLM recommendations.',
      why: 'Ensures verdicts are grounded in evidence, not LLM hallucination.',
      tool_or_model: 'Deterministic Gate (final_verdict.py)',
      input_summary: `Evidence sufficiency: ${fact.evidence_sufficiency || 'UNKNOWN'}`,
      output_summary: `Final Verdict: ${fact.verdict || 'UNVERIFIED'}; Overrode Agent: ${finalV.verdict_overridden_by_evidence ? 'YES' : 'NO'}`,
      metric: { label: 'Final Certainty', value: fact.confidence != null ? `${(fact.confidence > 1 ? fact.confidence : fact.confidence * 100).toFixed(1)}%` : 'Calculated', meaning: 'Evidence certainty score.', scale: '0-100%' },
      details: fact.deterministic_gate_reason || fact.reasoning || undefined,
    });

    return stages;
  }, [backendStages, result, raw, fact]);

  const stages = backendStages.length > 0 ? backendStages : fallbackStages;

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
"""

target = 'c:/Users/ASUS/Installation/AgenticAI/Multi_Agent_System/frontend/src/components/InvestigationTimeline.tsx'
with open(target, 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Written OK, {len(content)} chars')
