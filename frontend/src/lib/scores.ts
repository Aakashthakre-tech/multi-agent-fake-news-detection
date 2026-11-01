/**
 * scores.ts
 * ========
 * Formatting for the numeric scores the backend returns.
 *
 * The response mixes two kinds of number, and they must not be shown the same
 * way:
 *
 *   - **ratios** (`source_agreement`, `fusion_score`) are normalized to 0..1
 *     and are percentages.
 *   - **counts** (`entailment_score`, `contradiction_score`) are the number of
 *     strong evidence chunks. They are NOT percentages: a count of 6 must
 *     never be rendered as "600.0%".
 *
 * The kind is declared explicitly here, in one place, so no two components
 * can disagree about how a score reads.
 */

export type ScoreKind = 'ratio' | 'count';

/** How each score in `factual_verification` must be read. */
export const SCORE_KINDS: Record<string, ScoreKind> = {
  source_agreement: 'ratio',
  fusion_score: 'ratio',
  entailment_score: 'count',
  contradiction_score: 'count',
};

export function scoreKindFor(field: string): ScoreKind {
  return SCORE_KINDS[field] ?? 'ratio';
}

/**
 * Render a score.
 *
 * A ratio is only shown as a percentage when it really is a normalized ratio
 * (0..1); anything outside that range is shown as the number it is.
 */
export function formatScore(
  value: number | null | undefined,
  kind: ScoreKind
): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  if (kind === 'count') {
    const n = Math.round(value);
    return `${n} chunk${n === 1 ? '' : 's'}`;
  }
  if (value < 0 || value > 1) {
    // Not a normalized ratio: show the number rather than inventing a scale.
    return value.toFixed(1);
  }
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * The label for the final confidence.
 *
 * The number means different things depending on the verdict, so the label
 * follows `confidence_basis`:
 *   - `verification_coverage` - the share of the retrieved material that could
 *     be read. It is NOT a probability that the claim is true.
 *   - `evidence_strength` - the strength of the evidence behind a settled
 *     verdict.
 */
export function confidenceLabel(basis?: string | null): string {
  return basis === 'verification_coverage' ? 'Evidence Coverage' : 'Evidence Certainty';
}

export function confidenceNote(basis?: string | null): string | null {
  if (basis === 'verification_coverage') {
    return 'Share of the retrieved material that could be read. Not a probability that the claim is true.';
  }
  return null;
}
