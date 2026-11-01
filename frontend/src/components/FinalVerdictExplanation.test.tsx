/**
 * FinalVerdictExplanation.test.tsx
 * ================================
 * Render tests for the authoritative final-verdict panel.
 *
 * The panel is the UI's final-verdict section, so it must display the SAME
 * values the backend adjudicated: the final verdict, the final confidence with
 * its correct label, the evidence sufficiency, the independent source counts,
 * the evidence scores, the permitted verdicts, and the processing status.
 *
 * The fixture mirrors a real `/verify/text` response field for field.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FinalVerdictExplanation } from './FinalVerdictExplanation';
import { UnifiedVerificationResponse } from '../types';

function makeResponse(overrides: Partial<UnifiedVerificationResponse> = {}): UnifiedVerificationResponse {
  const fact = {
    verdict: 'REAL' as const,
    confidence: 0.659,
    confidence_basis: 'evidence_strength',
    evidence_sufficiency: 'strong_support',
    independent_supporting: 2,
    independent_contradicting: 0,
    contradiction_score: 0.0,
    entailment_score: 0.9133,
    source_agreement: 0.5,
    fusion_score: 0.72,
    total_sources: 7,
    permitted_verdicts: ['REAL', 'UNVERIFIED'],
    reasoning: 'Two independent publishers directly support the claim.',
    deterministic_gate_reason:
      '2 independent publishers, 2 of them primary publishers with retrieved ' +
      'article text, directly support the claim.',
    agent_verdict: 'REAL',
    critic_verdict: 'REAL',
    agents_agree: true,
    verdict_overridden_by_evidence: false,
    agent_confidence: 95.0,
    critic_confidence: 93.0,
  };

  return {
    modality: 'text',
    status: 'success',
    factual_verification: fact as UnifiedVerificationResponse['factual_verification'],
    synthetic_media_detection: null,
    visual_classification: null,
    transcription: null,
    primary_claim: "NASA's Apollo 11 mission landed humans on the Moon.",
    consistency_analysis: null,
    case_summary: null,
    pipeline_stages: [],
    modality_specific: null,
    research_report: null,
    raw: { final_verdict: {} },
    error: null,
    warnings: [],
    ...overrides,
  } as UnifiedVerificationResponse;
}

describe('FinalVerdictExplanation - the authoritative final verdict', () => {
  it('displays the final adjudicated verdict', () => {
    render(<FinalVerdictExplanation result={makeResponse()} />);
    // The verdict appears in the banner and in the permitted-verdict chips.
    expect(screen.getAllByText('REAL').length).toBeGreaterThan(0);
  });

  it('labels a settled verdict confidence as evidence certainty', () => {
    render(<FinalVerdictExplanation result={makeResponse()} />);
    expect(screen.getByText('Evidence Certainty')).toBeTruthy();
    expect(screen.queryByText('Evidence Coverage')).toBeNull();
  });

  it('labels an UNVERIFIED confidence as evidence coverage, not certainty', () => {
    // For an UNVERIFIED result the number is the share of retrieved material
    // that could be read. Labelling it "Evidence Certainty" would read as
    // certainty about the claim.
    const response = makeResponse({
      factual_verification: {
        ...makeResponse().factual_verification,
        verdict: 'UNVERIFIED',
        confidence: 1.0,
        confidence_basis: 'verification_coverage',
        evidence_sufficiency: 'context_only',
        independent_supporting: 0,
        independent_contradicting: 0,
        entailment_score: 0.0,
        source_agreement: 0.0,
        fusion_score: 0.0,
        total_sources: 9,
        permitted_verdicts: ['UNVERIFIED'],
      } as UnifiedVerificationResponse['factual_verification'],
    });
    const { container } = render(<FinalVerdictExplanation result={response} />);
    expect(screen.getByText('Evidence Coverage')).toBeTruthy();
    expect(screen.queryByText('Evidence Certainty')).toBeNull();
    expect(container.textContent).toMatch(/Not a probability that the claim is true/i);
  });

  it('displays the final-assessment fields', () => {
    const { container } = render(<FinalVerdictExplanation result={makeResponse()} />);
    const text = container.textContent || '';
    expect(text).toContain('Evidence Sufficiency');
    expect(text).toContain('strong_support');
    expect(text).toContain('Source Agreement');
    expect(text).toContain('Fusion Score');
    expect(text).toContain('Entailment Score');
    expect(text).toContain('Contradiction Score');
    expect(text).toContain('Supporting Sources');
    expect(text).toContain('Contradicting Sources');
  });

  it('shows entailment and contradiction scores as counts, not percentages', () => {
    // entailment_score and contradiction_score are COUNTS of strong evidence
    // chunks. A count of 6 shown as "600.0%" would be a false statement.
    const response = makeResponse({
      factual_verification: {
        ...makeResponse().factual_verification,
        entailment_score: 6,
        contradiction_score: 2,
      } as UnifiedVerificationResponse['factual_verification'],
    });
    const { container } = render(<FinalVerdictExplanation result={response} />);
    const text = container.textContent || '';

    expect(text).toContain('6 chunks');
    expect(text).toContain('2 chunks');
    expect(text).not.toContain('600.0%');
    expect(text).not.toContain('200.0%');
  });

  it('shows the ratios as percentages', () => {
    const { container } = render(<FinalVerdictExplanation result={makeResponse()} />);
    const text = container.textContent || '';
    expect(text).toContain('50.0%'); // source agreement
    expect(text).toContain('72.0%'); // fusion score
  });

  it('displays the verdicts the evidence floor permitted', () => {
    const { container } = render(<FinalVerdictExplanation result={makeResponse()} />);
    expect(container.textContent).toMatch(/Verdicts permitted by the evidence floor/i);
  });

  it('displays the deterministic gate reason', () => {
    const { container } = render(<FinalVerdictExplanation result={makeResponse()} />);
    expect(container.textContent).toContain('2 independent publishers');
  });

  it('surfaces a partial verification instead of reading as finished', () => {
    const { container } = render(
      <FinalVerdictExplanation result={makeResponse({ status: 'partial' })} />
    );
    expect(container.textContent).toMatch(/Partial verification/i);
    expect(container.textContent).toContain('PARTIAL');
  });

  it('does not show a partial banner for a successful result', () => {
    const { container } = render(
      <FinalVerdictExplanation result={makeResponse({ status: 'success' })} />
    );
    expect(container.textContent).not.toMatch(/Partial verification/i);
  });

  it('renders without crashing when optional fields are missing', () => {
    const response = makeResponse({
      factual_verification: {
        verdict: 'UNVERIFIED',
        confidence: null,
        evidence_sufficiency: null,
        independent_supporting: 0,
        independent_contradicting: 0,
      } as UnifiedVerificationResponse['factual_verification'],
    });
    const { container } = render(<FinalVerdictExplanation result={response} />);
    expect(container.textContent).toContain('UNVERIFIED');
    expect(container.textContent).toContain('N/A');
  });
});
