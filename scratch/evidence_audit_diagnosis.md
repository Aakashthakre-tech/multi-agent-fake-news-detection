# Evidence-pipeline audit — diagnosis

Scope: `evidence_analyzer.py`, `evidence_brief.py`, `evidence_fusion.py`,
`final_verdict.py`, `claim_profile.py`, `evidence_matcher.py`,
`nli_checker.py`, `evidence_scorer.py`, `source_reliability.py`, `pipeline.py`.

All findings below are reproduced by `scratch/repro_evidence_defects.py`
(synthetic fixtures — they show what the code does, not real-world accuracy).

---

## 1. Confirmed defects (reproduced)

### D1 — Snippet-only evidence can refute a claim (P0)
`final_verdict.deterministic_gate` requires `primary_supporting_groups >= 1` for
REAL but applies **no article requirement at all** to the contradiction path.
Two `search_snippet` chunks from two publishers → `permitted_verdicts` includes
FAKE → final verdict FAKE.
`evidence_analyzer.classify_evidence` never classifies a snippet as
CONTRADICTION, so the analyzer is safe; the *verdict* layer re-derives the
verdict from raw records and loses that protection.
**Fix:** symmetric article/primary requirement for contradiction.

### D2 — "primary reporting" was inferred from "we scraped it" (P0)
`evidence_fusion._summarise_source` sets the group's `is_primary` from
`evidence_type == "scraped_article"`, discarding the real tier-based
`is_primary_source` that `source_reliability.get_source_profile` already
computes and that `evidence_analyzer` already stores on every record.
Two `weak/blog` sources → `primary_supporting_groups = 2` → REAL permitted.
**Fix:** use the recorded `is_primary_source`.

### D3 — MISLEADING is permitted with no evidence-backed correction (P0)
The support branch permits `{"REAL", "MISLEADING", "UNVERIFIED"}` whenever
support is sufficient. Nothing checks that any evidence actually corrects a
material claim detail. `evidence_analyzer` already records a per-chunk
`debunking` flag and classifies date-conflicting debunking as CONTRADICTION.
**Fix:** permit MISLEADING only when an explicit correction is present in the
evidence.

### D4 — Unrelated sources merge when `source_id` is missing (P0)
`evidence_fusion.calculate_source_agreement` groups by
`result.get("source_id")`. Records with no `source_id` all collapse into one
bucket: three different publishers → `total_sources = 1`,
`independence_groups = 1`, `independent_supporting = 1`.
**Fix:** key sources by canonical URL; `source_id` becomes an attribute, not
the key. Records with no resolvable identity are marked unresolved and
excluded from independent corroboration but kept for diagnostics.

### D5 — A reused `source_id` merges different URLs (P0)
Two records with `source_id = 1` but different URLs collapse into one source.
**Fix:** same as D4.

### D6 — A record with no resolvable independence group merges into one group (P0)
`_independence_group` returns `""` for a record with neither URL nor domain, so
every such record lands in the `""` group and is counted as one publisher.
**Fix:** unresolved groups are excluded from independent corroboration.

### D7 — `supporting_evidence` and `contradicting_evidence` are mislabelled (P1)
`final_verdict._primary_evidence` selects `wanted = "SUPPORT" if verdict in
("REAL","MISLEADING") else "CONTRADICTION"`. For an UNVERIFIED verdict both
lists are populated with CONTRADICTION chunks, so the API reports contradiction
passages as supporting evidence.
**Fix:** the two lists are selected by class, never by verdict.

### D8 — Agent verdict extraction can pick an incidental label (P1)
`final_verdict.extract_verdict` falls back to the first bare
`\b(REAL|FAKE|MISLEADING|UNVERIFIED)\b` anywhere in the text, so prose such as
"It is not REAL to say the event happened; the record suggests FAKE" yields
REAL. The agents are instructed to answer with a `Verdict:` / `Recommended
Verdict:` heading, so the fallback is unnecessary and unsafe.
**Fix:** labelled match (last one wins) → standalone-line match → `None`.

### D9 — UNVERIFIED confidence is a high heuristic constant (P1)
`calculate_final_confidence` returns 70 / 55 / 60 for UNVERIFIED. "70% "
next to UNVERIFIED reads as "70% sure", which is not what was measured and is
not a calibrated probability.
**Fix:** report UNVERIFIED confidence as *verification coverage*, label it
explicitly, and keep the numeric field for compatibility.

### D10 — Group-level conflict resolution ignores the source-level margin (P1)
`_summarise_source` treats a near-tie (within 0.10) as `neutral`, but the
group aggregation picks the higher score with no margin, so a group whose
sources disagree becomes a decisive vote.
**Fix:** use the same margin at group level.

---

## 2. Plausible defects requiring reproduction

- **P1 — location aliasing.** `location_terms` contains the full place and its
  word parts, so "New Delhi" matches a chunk saying "Delhi". Common
  abbreviations ("NYC", "NCR", "UK") and transliterations do not match, which
  can reject genuine evidence as `location_conflict`.
- **P1 — publication date vs event date.** `check_date_compatibility` already
  separates the two; needs a real trace to confirm the reported case is
  covered.
- **P1 — correction / update reporting.** A later article that corrects an
  earlier report may be classified CONTRADICTION rather than "correction",
  because `debunking` is a regex over correction language only.
- **P1 — adjacent-chunk splitting.** A fact split across two chunks may fall
  below `detail_coverage >= 0.30` for either chunk.

## 3. Existing safeguards that must remain intact

- `classify_evidence`: a snippet can never be SUPPORT/CONTRADICTION; a
  non-primary source can never SUPPORT/CONTRADICT on its own; SUPPORT needs
  exact date + strong event + location + coverage + NLI entailment;
  CONTRADICTION needs a date match or an explicit debunking correction;
  similarity alone never classifies; NLI failure never classifies.
- `evidence_scorer.apply_class_ceiling`: snippet ≤ 0.45, search_extract ≤ 0.80,
  RELATED_CONTEXT ≤ 0.25, NON_EVIDENCE ≤ 0.02.
- `evidence_fusion`: many chunks from one article count once; many pages of one
  publisher count once; one independent voice caps fusion at 0.62; genuine
  disagreement caps fusion at 0.50.
- `final_verdict`: the gate is permissive-only — agents may narrow it, never
  widen it; UNVERIFIED is always available.
- `pipeline`: only independently corroborated primary publishers count;
  syndicated copies are merged by `is_near_duplicate`.

## 4. Files and functions to change

| File | Function | Change |
|---|---|---|
| `evidence_fusion.py` | `_canonical_url`, `_source_key` (new) | stable source identity |
| `evidence_fusion.py` | `_independence_group` | unresolved groups flagged, not merged |
| `evidence_fusion.py` | `_summarise_source` | real `is_primary`, snippet/Article flags, correction flag, identity flag |
| `evidence_fusion.py` | `calculate_source_agreement` | new grouping, primary/Article counts both ways, correction groups, unresolved count, consistent margin |
| `evidence_fusion.py` | `calculate_fusion_score` | use the new article flag |
| `final_verdict.py` | `deterministic_gate` | symmetric article requirement, MISLEADING requires a correction, accurate reasons |
| `final_verdict.py` | `extract_verdict` | no incidental-label fallback |
| `final_verdict.py` | `_primary_evidence` | select by class, not by verdict |
| `final_verdict.py` | `calculate_final_confidence`, `generate_final_verdict` | confidence basis + coverage |
| `evidence_brief.py` | `build_evidence_brief` | report the real state, not a context-only default |
| `api/server.py`, `frontend/src/types/index.ts`, report panel | — | expose the new diagnostics honestly |
