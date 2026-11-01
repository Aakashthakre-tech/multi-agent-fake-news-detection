import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { UnifiedReport } from './UnifiedReport';
import { UnifiedVerificationResponse } from '../types';

function makeFact(overrides: any = {}) {
  const base = {
    verdict: 'REAL' as const,
    confidence: 0.83,
    confidence_basis: 'evidence_strength',
    evidence_sufficiency: 'strong_support',
    reasoning: 'Two independent publishers directly support the claim.',
    supporting_evidence: [],
    contradicting_evidence: [],
  };
  return { ...base, ...overrides };
}

function makeResponse(fact: any): UnifiedVerificationResponse {
  return {
    modality: 'text',
    status: 'success',
    factual_verification: fact,
    synthetic_media_detection: null,
    visual_classification: null,
    transcription: null,
    primary_claim: "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
    consistency_analysis: null,
    case_summary: null,
    pipeline_stages: null,
    modality_specific: null,
    research_report: null,
    raw: {},
    error: null,
    warnings: [],
  } as UnifiedVerificationResponse;
}

describe('UnifiedReport final verdict summary', () => {
  it('renders the final assessment header', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'REAL' }))}
        modality="text"
        claimText="Apollo 11 landed on the Moon in 1969."
        selectedFile={null}
      />
    );
    expect(screen.getByText('TRUTHLENS AI — FINAL ASSESSMENT').closest('div')).toBeTruthy();
    expect(screen.getByText(/TRUTHLENS AI — FINAL ASSESSMENT/)).toBeInTheDocument();
  });

  it('renders the submitted claim', () => {
    const { container } = render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'REAL' }))}
        modality="text"
        claimText="Apollo 11 landed on the Moon in 1969."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/Claim/)).toBeInTheDocument();
    expect(container.textContent).toContain("NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969.");
  });

  it('renders the backend verdict badge', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'REAL' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/REAL/)).toBeInTheDocument();
  });

  it('renders MISLEADING and UNVERIFIED verdicts without converting them', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'MISLEADING' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/MISLEADING/)).toBeInTheDocument();

    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'UNVERIFIED' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/UNVERIFIED/)).toBeInTheDocument();
  });

  it('renders AI confidence as a percentage for an evidence_strength basis', () => {
    const { container } = render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'REAL', confidence: 0.6 }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/AI Confidence/)).toBeInTheDocument();
    expect(container.textContent).toContain('60.0%');
  });

  it('shows Confidence unavailable when no valid claim probability exists', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'FAKE', confidence_basis: 'verification_coverage' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/AI Confidence/)).toBeInTheDocument();
    expect(screen.getByText(/Confidence unavailable/)).toBeInTheDocument();
    expect(screen.queryByText(/\d+%/)).not.toBeInTheDocument();
  });

  it('shows Confidence unavailable when confidence is null', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'UNVERIFIED', confidence: null }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/Confidence unavailable/)).toBeInTheDocument();
  });

  it('renders the backend final-assessment explanation as the summary', () => {
    const reasoning =
      'Available sources report fires in Delhi on the specified date. ' +
      'The retrieved evidence does not sufficiently establish the exact event. ' +
      'Therefore the claim cannot currently be confirmed or refuted. ' +
      'Extra sentence that must not appear.';
    const { container } = render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'UNVERIFIED', reasoning }))}
        modality="text"
        claimText="Major fire breaks out at a commercial building in New Delhi on 12 Aug 2024."
        selectedFile={null}
      />
    );
    const text = container.textContent || '';
    expect(text).toContain('Available sources report fires in Delhi on the specified date.');
    expect(text).toContain('The retrieved evidence does not sufficiently establish the exact event.');
    expect(text).toContain('Therefore the claim cannot currently be confirmed or refuted.');
    expect(text).not.toContain('Extra sentence that must not appear.');
  });

  it('falls back to a verdict-based summary when the backend supplies no reasoning', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'UNVERIFIED', reasoning: '' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/insufficient to confirm or refute this claim/)).toBeInTheDocument();
  });

  it('renders Image Origin Analysis for an image result with probabilities', () => {
    const imageResult = {
      ...makeResponse(makeFact({ verdict: 'FAKE' })),
      modality: 'image' as const,
      synthetic_media_detection: {
        available: true,
        signal: 'AI_GENERATED',
        probabilities: { ai_probability: 0.432, real_probability: 0.568 },
      },
    } as unknown as UnifiedVerificationResponse;

    const { container } = render(
      <UnifiedReport
        result={imageResult}
        modality="image"
        claimText="Claim."
        selectedFile={null}
      />
    );
    const text = container.textContent || '';
    expect(screen.getByText(/Image Origin Analysis/)).toBeInTheDocument();
    expect(screen.getByText(/AI-generated probability/)).toBeInTheDocument();
    expect(text).toContain('43.2%');
    expect(screen.getByText(/Real-image probability/)).toBeInTheDocument();
    expect(text).toContain('56.8%');
    expect(screen.getByText(/Origin assessment/)).toBeInTheDocument();
    expect(screen.getByText(/AI_GENERATED/)).toBeInTheDocument();
  });

  it('hides Image Origin Analysis when no image-origin data is available', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'REAL' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.queryByText(/Image Origin Analysis/)).not.toBeInTheDocument();
  });

  it('removes the technical evidence explanation from the result screen', () => {
    render(
      <UnifiedReport
        result={makeResponse(makeFact({ verdict: 'REAL' }))}
        modality="text"
        claimText="Claim."
        selectedFile={null}
      />
    );
    expect(screen.queryByText(/Key Evidence/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Evidence Status/)).not.toBeInTheDocument();
    expect(screen.queryByText(/SUFFICIENT|LIMITED|INSUFFICIENT/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Fusion Score/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Source Agreement/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Entailment Score/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Contradiction Score/)).not.toBeInTheDocument();
  });

  it('renders error state', () => {
    render(
      <UnifiedReport
        result={{
          modality: 'text',
          status: 'error',
          factual_verification: makeFact({ verdict: 'UNVERIFIED' }),
          synthetic_media_detection: null,
          visual_classification: null,
          transcription: null,
          primary_claim: null,
          consistency_analysis: null,
          case_summary: null,
          pipeline_stages: null,
          modality_specific: null,
          research_report: null,
          raw: {},
          error: 'Tavily API rate limit exceeded.',
          warnings: [],
        } as UnifiedVerificationResponse}
        modality="text"
        claimText="Test claim."
        selectedFile={null}
      />
    );
    expect(screen.getByText(/Verification Failed/)).toBeInTheDocument();
    expect(screen.getByText(/Tavily API rate limit exceeded/)).toBeInTheDocument();
  });
});
