/**
 * scores.test.tsx
 * ==============
 * Focused tests for the score and confidence labels.
 *
 * The backend mixes two kinds of number in the same response:
 *   - ratios (`source_agreement`, `fusion_score`) normalized to 0..1;
 *   - counts (`entailment_score`, `contradiction_score`) of strong evidence
 *     chunks.
 *
 * A count of 6 must never be rendered as "600.0%", and a coverage figure
 * must never be presented as a probability that the claim is true.
 */
import { describe, it, expect } from 'vitest';
import {
  confidenceLabel,
  confidenceNote,
  formatScore,
  scoreKindFor,
} from '../lib/scores';

describe('score kinds', () => {
  it('knows which backend fields are ratios and which are counts', () => {
    expect(scoreKindFor('source_agreement')).toBe('ratio');
    expect(scoreKindFor('fusion_score')).toBe('ratio');
    expect(scoreKindFor('entailment_score')).toBe('count');
    expect(scoreKindFor('contradiction_score')).toBe('count');
  });
});

describe('formatScore', () => {
  it('renders a ratio as a percentage', () => {
    expect(formatScore(0.6111, 'ratio')).toBe('61.1%');
    expect(formatScore(0, 'ratio')).toBe('0.0%');
    expect(formatScore(1, 'ratio')).toBe('100.0%');
  });

  it('renders a count as a count, never as a percentage', () => {
    expect(formatScore(6, 'count')).toBe('6 chunks');
    expect(formatScore(1, 'count')).toBe('1 chunk');
    expect(formatScore(0, 'count')).toBe('0 chunks');
    // The bug this prevents: a count of 6 shown as "600.0%".
    expect(formatScore(6, 'count')).not.toContain('%');
  });

  it('never invents a percentage for a value outside the ratio range', () => {
    expect(formatScore(4.2, 'ratio')).toBe('4.2');
    expect(formatScore(-1, 'ratio')).toBe('-1.0');
  });

  it('reports a missing value as unavailable', () => {
    expect(formatScore(null, 'ratio')).toBe('N/A');
    expect(formatScore(undefined, 'count')).toBe('N/A');
    expect(formatScore(NaN, 'ratio')).toBe('N/A');
  });
});

describe('confidence labels', () => {
  it('follows the confidence basis', () => {
    expect(confidenceLabel('verification_coverage')).toBe('Evidence Coverage');
    expect(confidenceLabel('evidence_strength')).toBe('Evidence Certainty');
    expect(confidenceLabel(undefined)).toBe('Evidence Certainty');
  });

  it('explains that coverage is not a probability', () => {
    expect(confidenceNote('verification_coverage')).toMatch(
      /not a probability that the claim is true/i
    );
    expect(confidenceNote('evidence_strength')).toBeNull();
  });
});
