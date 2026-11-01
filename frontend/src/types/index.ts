// TruthLensAI TypeScript Domain Types

export type ModalityType = 'text' | 'image' | 'video' | 'audio';

export type FactualVerdict = 'REAL' | 'FAKE' | 'MISLEADING' | 'UNVERIFIED' | 'INCONCLUSIVE' | 'SUPPORTED' | 'CONTRADICTED' | null;

export type SyntheticMediaSignal = 'AI_GENERATED' | 'REAL' | 'UNCERTAIN' | 'UNAVAILABLE' | null;

// ---------------------------------------------------------------------------
// Research report (point-wise, evidence-grounded) — built server-side by
// research_report.py from the real pipeline state.
// ---------------------------------------------------------------------------

export type ResearchFindingStatus = 'SUPPORTED' | 'CONTRADICTED' | 'CONTEXT' | 'NON_EVIDENCE' | string;

export interface ResearchFinding {
  title: string;
  explanation: string;
  claim_text?: string | null;
  status: ResearchFindingStatus;
  evidence_strength?: number | null;
  strength_label?: string | null;
  source_name?: string | null;
  source_domain?: string | null;
  source_url?: string | null;
  source_tier?: string | null;
  source_reliability?: number | null;
  excerpt?: string | null;
  excerpt_kind?: string | null;
  date_status?: string | null;
  event_match?: string | null;
  nli_label?: string | null;
  nli_score?: number | null;
  independence_group?: string | null;
  evidence_class?: string | null;
  evidence_type?: string | null;
  search_query?: string | null;
}

export interface PolicyDiagnostics {
  /**
   * False when the deterministic evidence gate never evaluated this result
   * (e.g. a modality with no verifiable claim to research).
   */
  gate_ran: boolean;
  /**
   * False when the evidence policy counters were missing or malformed, so the
   * gate could not apply its policy and restricted the result to UNVERIFIED.
   * This is an internal evidence-accounting issue, not proof that the claim
   * is false.
   */
  policy_data_valid: boolean | null;
  /** The missing or invalid policy fields, when any. */
  policy_data_errors: string[];
  /** True when the policy data forced the gate to restrict the verdict. */
  verdict_restricted: boolean;
  /** Plain-language explanation of the state. */
  explanation: string | null;
}

export interface AgentStatus {
  verification_agent: string;
  critic_agent: string;
  /** True when an agent stage failed, so the result is a partial verification. */
  partial: boolean;
  errors: { stage?: string | null; error?: string | null }[];
}

export interface ResearchExecutiveSummary {
  input_claim?: string | null;
  modality?: string | null;
  verdict?: string | null;
  confidence?: number | null;
  /** What `confidence` measures: 'evidence_strength' or 'verification_coverage'. */
  confidence_basis?: string | null;
  confidence_basis_note?: string | null;
  confidence_basis_explanation?: string | null;
  /** The deterministic gate's own policy-accounting diagnostics. */
  policy_diagnostics?: PolicyDiagnostics | null;
  /** Which agent stages ran, and which failed. */
  agent_status?: AgentStatus | null;
  attempted?: string | null;
  key_findings: string[];
  why_this_verdict?: string | null;
  deterministic_gate_reason?: string | null;
  agent_narrative?: string | null;
  critic_review?: string | null;
  limitations: string[];
}

export interface ResearchSearchQuery {
  label?: string | null;
  query: string;
  intent?: string | null;
  topic?: string | null;
}

export interface ResearchScrapeFailure {
  url?: string | null;
  domain?: string | null;
  title?: string | null;
  reason?: string | null;
  fallback_used?: string | boolean | null;
}

export interface ResearchMethodologyStage {
  name: string;
  status: string;
  detail?: string | null;
}

export interface ResearchMethodology {
  research_skipped: boolean;
  research_skipped_reason?: string | null;
  search_queries: ResearchSearchQuery[];
  search_errors: string[];
  sources_discovered?: number;
  sources_retrieved: number;
  articles_scraped: number;
  scrapes_failed: ResearchScrapeFailure[];
  syndicated_sources_merged?: number;
  evidence_passages?: number;
  nli_evaluated?: number;
  // --- evidence-policy counters (real, from the fusion layer) ----------
  /** Independent publisher groups the retrieved sources collapsed into. */
  independent_publishers?: number;
  /** Supporting groups that are primary reporting with retrieved article text. */
  primary_supporting_groups?: number;
  primary_contradicting_groups?: number;
  article_supporting_groups?: number;
  article_contradicting_groups?: number;
  /** Groups containing an explicit correction of a material claim detail. */
  correction_groups?: number;
  snippet_only_sources?: number;
  article_sources?: number;
  /** Sources whose publisher identity could not be established. */
  unresolved_identity_sources?: number;
  /** Share of the retrieved pool that could be read and classified (0..1). */
  verification_coverage?: number;
  stages: ResearchMethodologyStage[];
}

export interface ResearchClaimResult {
  claim_text?: string | null;
  verdict?: string | null;
  confidence?: number | null;
  evidence_sufficiency?: string | null;
  reasoning?: string | null;
  supporting_evidence: any[];
  contradicting_evidence: any[];
  source_name?: string | null;
  source_url?: string | null;
  timestamps?: any;
  frame_references?: number[] | null;
  source_modality?: string | null;
  selection_reason?: string | null;
  priority_score?: number | null;
  limitations: string[];
}

export interface ResearchEvidenceGap {
  question: string;
  status: string;
  detail?: string | null;
  what_would_help?: string | null;
}

/**
 * A single adjudicated claim, as returned by the backend for a given
 * modality (candidate claims for video/audio, the input claim for text).
 */
export interface ClaimRecord {
  claim_id?: string;
  claim?: string | null;
  claim_text?: string | null;
  original_text?: string | null;
  claim_verdict?: string | null;
  verdict?: string | null;
  confidence?: number | null;
  evidence_sufficiency?: string | null;
  selection_reason?: string | null;
  selection_status?: string | null;
  source_modality?: string | null;
  source_provenance?: string | null;
  priority_score?: number | null;
  timestamps?: [number, number] | number[] | null;
  frame_references?: number[] | null;
  is_factual?: boolean;
  is_verifiable?: boolean;
  reasoning?: string | null;
  source_name?: string | null;
  source_url?: string | null;
  supporting_evidence?: any[];
  contradicting_evidence?: any[];
  evidence_analysis?: any[];
  source_data?: any[];
  limitations?: string[];
}

export interface ResearchReport {
  modality: string;
  executive_summary: ResearchExecutiveSummary;
  findings: {
    supporting?: ResearchFinding[];
    contradicting?: ResearchFinding[];
    contextual?: ResearchFinding[];
    unresolved?: ResearchFinding[];
  };
  claims: ResearchClaimResult[];
  methodology: ResearchMethodology;
  evidence_gaps: ResearchEvidenceGap[];
}

export interface FactualVerificationDetails {
  verdict: FactualVerdict;
  confidence: number | null;
  /**
   * What `confidence` actually measures.
   *   'evidence_strength'    - strength of the evidence behind a settled verdict
   *   'verification_coverage' - share of retrieved material that could be read
   *                            and classified. Used when the verdict is
   *                            UNVERIFIED. It is NOT a probability that the
   *                            claim is true or false.
   */
  confidence_basis?: 'evidence_strength' | 'verification_coverage' | string | null;
  evidence_sufficiency: 'SUFFICIENT' | 'INSUFFICIENT' | string | null;
  independent_supporting: number;
  independent_contradicting: number;
  contradiction_score: number | null;
  /** Strength of the entailment evidence behind the verdict. */
  entailment_score?: number | null;
  source_agreement: number | null;
  fusion_score: number | null;
  reasoning?: string | null;
  permitted_verdicts?: string[] | null;
  deterministic_gate_reason?: string | null;
  total_sources?: number;
  supporting_sources?: number;
  contradicting_sources?: number;
  context_chunks?: number;
  /** Primary evidence passages supporting the verdict (from final_verdict.py). */
  supporting_evidence?: any[];
  /** Primary evidence passages contradicting the verdict (from final_verdict.py). */
  contradicting_evidence?: any[];
  agent_verdict?: string | null;
  critic_verdict?: string | null;
  agents_agree?: boolean | null;
  verdict_overridden_by_evidence?: boolean | null;
  agent_confidence?: number | null;
  critic_confidence?: number | null;
  primary_supporting_groups?: number;
  // --- evidence-policy diagnostics -------------------------------------
  /** Publishers whose article text was actually read (not a search snippet). */
  primary_contradicting_groups?: number;
  article_supporting_groups?: number;
  article_contradicting_groups?: number;
  /** Publishers containing an explicit correction of a material claim detail. */
  correction_groups?: number;
  /** Share of the retrieved pool that could be read and classified (0..1). */
  verification_coverage?: number | null;
  /** Sources whose publisher identity could not be established. */
  unresolved_identity_sources?: number;
  /**
   * False when the evidence policy counters were missing or malformed, so the
   * deterministic gate restricted the result to UNVERIFIED. This is an
   * internal evidence-accounting issue, not proof that the claim is false.
   */
  policy_data_valid?: boolean | null;
  /** The missing or invalid policy fields, when any. */
  policy_data_errors?: string[];

  /** Google Fact Check API result, if the API was queried. */
  google_fact_check?: {
    status: 'success' | 'error' | null;
    error_category: string | null;
    error_message: string | null;
    claims: Array<{
      text: string;
      claimDate?: string | null;
      claimant?: string | null;
      claimReview: Array<{
        publisher: { name: string };
        textualRating: string;
        url: string;
        title?: string | null;
        reviewDate?: string | null;
        languageCode?: string | null;
      }>;
    }>;
  } | null;

  /** Independent LLM second opinion on the claim and evidence package. */
  llm_second_opinion?: LLMSecondOpinion | null;

  /** Independent direct-LLM analysis of the original claim (knowledge-based). */
  independent_llm_analysis?: IndependentLLMAnalysis | null;
}

export interface LLMSecondOpinion {
  verdict: 'REAL' | 'FAKE' | 'MISLEADING' | 'UNVERIFIED' | null;
  confidence: number | null; // 0..100, only when a valid verdict exists
  claim_summary: string;
  explanation: string;
  source_credibility: string;
  evidence_sufficiency: string;
  supporting_evidence: Array<{ url: string; domain: string; excerpt: string }>;
  contradicting_evidence: Array<{ url: string; domain: string; excerpt: string }>;
  uncertainty: string;
  limitations: string[];
  raw_output: string;
  error: string | null;
  timeout: boolean;
  api_failure: boolean;
  no_evidence: boolean;
  comparison: {
    existing_verdict: string | null;
    llm_verdict: 'REAL' | 'FAKE' | 'MISLEADING' | 'UNVERIFIED' | null;
    agreement: boolean | null;
    disagreement: boolean | null;
    gate_enforced: boolean;
    agreement_with_system: string | null;
    comparison_summary: string;
  };
}

export interface IndependentLLMAnalysis {
  verdict: 'REAL' | 'FAKE' | 'MISLEADING' | 'UNVERIFIED' | null;
  confidence: number | null; // 0..100, only when a valid verdict exists
  claim_summary: string;
  reasoning: string;
  extracted_text: string | null; // concise text already extracted from media
  live_verification_available: boolean | null;
  supporting_evidence: Array<{ url: string; domain: string; excerpt: string }>;
  contradicting_evidence: Array<{ url: string; domain: string; excerpt: string }>;
  uncertainty: string;
  limitations: string[];
  raw_output: string;
  error: string | null;
  timeout: boolean;
  api_failure: boolean;
  comparison: {
    existing_verdict: string | null;
    llm_verdict: 'REAL' | 'FAKE' | 'MISLEADING' | 'UNVERIFIED' | null;
    agreement: boolean | null;
    disagreement: boolean | null;
    agreement_with_system: string | null;
    comparison_summary: string;
    live_verification_available: boolean | null;
  };
}

export interface EvidenceItem {
  source_id?: string | number;
  url: string;
  domain: string;
  title?: string;
  evidence_type: 'scraped_article' | 'search_extract' | 'search_snippet' | string;
  chunk: string;
  similarity?: number;
  nli_label?: string;
  nli_score?: number;
  nli_status?: string;
  raw_nli_label?: string;
  raw_nli_score?: number;
  source_reliability?: number;
  source_tier?: string;
  source_tier_reason?: string;
  independence_group?: string;
  is_primary_source?: boolean;
  is_rolling_page?: boolean;
  evidence_score?: number;
  evidence_class: 'SUPPORT' | 'CONTRADICTION' | 'RELATED_CONTEXT' | 'NON_EVIDENCE' | string;
  date_status?: string;
  date_reason?: string;
  publication_date?: {
    day?: number | null;
    month?: number | null;
    year?: number | null;
  } | string | null;
  event_match?: boolean | number;
  location_match?: boolean;
  location_conflict?: boolean;
  detail_coverage?: number;
  debunking?: boolean;
  event_signals?: string[];
}

export interface SourceCandidate {
  url: string;
  title?: string;
  domain?: string;
  score?: number;
  snippet?: string;
  content_kind?: string;
  published_date?: string;
  found_via?: string[];
  source_tier?: string;
  retrieval_class?: string;
  match_signals?: string;
  search_reliability?: number;
  scrape_status?: string;
}

export interface SyntheticMediaDetails {
  signal: SyntheticMediaSignal;
  confidence: number | null;
  model: string | null;
  available: boolean;
  warning?: string | null;
  probabilities?: {
    probability_synthetic?: number | null;
    probability_real?: number | null;
    ai_probability?: number | null;
    real_probability?: number | null;
    /** Always null for images: no manipulation detector is installed */
    manipulation_score?: number | null;
    [key: string]: any;
  } | null;
  /** 'NOT_ASSESSED' when no tampering detector is installed; absent for audio/video */
  manipulation_status?: 'NOT_ASSESSED' | 'UNAVAILABLE' | string | null;
  /** Human-readable explanation of why manipulation status is NOT_ASSESSED */
  manipulation_note?: string | null;
  /** The detector's own frozen operating point, as reported by the backend */
  operating_point?: {
    threshold?: number;
    uncertain_band?: number;
    temperature?: number;
    uncertain_low?: number;
    uncertain_high?: number;
  } | null;
  error?: string | null;
}

export interface VisualClassificationDetails {
  predicted_label: 'fake' | 'real' | string | null;
  confidence: number | null;
  experimental_warning?: string | null;
  available: boolean;
  probabilities?: {
    fake?: number;
    real?: number;
    [key: string]: any;
  } | null;
}

export interface CaseSummary {
  case_id: string;
  modality: ModalityType | string;
  filename?: string | null;
  file_size_bytes?: number | null;
  file_format?: string | null;
  media_metadata?: Record<string, any> | null;
  timestamp_utc: string;
  processing_time_sec?: number | null;
  validation_status: string;
  processing_status: string;
}

export interface PipelineStageMetric {
  label: string;
  value: string | number;
  meaning: string;
  scale: string;
}

export interface PipelineStageDetails {
  id: string;
  number: number;
  name: string;
  status: 'completed' | 'running' | 'skipped' | 'failed' | string;
  operation: string;
  why: string;
  tool_or_model: string;
  input_summary: string;
  output_summary: string;
  metric?: PipelineStageMetric | null;
  details?: string | null;
  duration_sec?: number | null;
  warnings?: string[] | null;
}

export interface CandidateClaim {
  claim_id: string;
  claim: string;
  original_text?: string;
  source_modality?: string;
  source_provenance?: string;
  priority_score?: number;
  selection_status?: 'selected' | 'rejected' | string;
  selection_reason?: string;
  timestamps?: [number, number] | number[];
  frame_references?: number[];
  is_verifiable?: boolean;
  is_factual?: boolean;
}

export interface DetectedObject {
  object: string;
  confidence: number;
  box?: number[];
}

export interface OcrResultItem {
  text: string;
  confidence?: number;
  box?: any;
}

export interface UnifiedVerificationResponse {
  modality: ModalityType;
  status: 'success' | 'error' | string;
  factual_verification: FactualVerificationDetails;
  synthetic_media_detection?: SyntheticMediaDetails | null;
  visual_classification?: VisualClassificationDetails | null;
  transcription?: string | null;
  transcription_segments?: TranscriptionSegment[] | null;
  primary_claim?: string | null;
  consistency_analysis?: {
    status?: string;
    description?: string;
    consistency_note?: string;
    extracted_claim?: string;
    detected_objects?: string[];
    [key: string]: any;
  } | null;
  case_summary?: CaseSummary | null;
  pipeline_stages?: PipelineStageDetails[] | null;
  modality_specific?: ModalitySpecificPayload | null;
  research_report?: ResearchReport | null;
  raw: Record<string, any>;
  error?: string | null;
  warnings: string[];
}

/** Whisper speech segment with timestamp. */
export interface TranscriptionSegment {
  id?: number;
  start?: number;
  end?: number;
  text: string;
  language?: string;
  avg_logprob?: number;
  no_speech_prob?: number;
}

/** Fused speech/OCR entry produced by video_pipeline.fuse_whisper_and_ocr. */
export interface FusedEntry {
  entry_id?: string;
  source?: string;
  text?: string;
  start_sec?: number;
  end_sec?: number;
  frame_references?: number[];
  on_screen_context?: string | null;
  disagreements?: string[];
  review_flags?: string[];
}

/** OCR observation on a single frame. */
export interface OcrObservation {
  frame_index?: number;
  timestamp_sec?: number;
  text?: string;
  confidence?: number;
  raw_items?: any[];
}

/** Video metadata captured by ffprobe. */
export interface VideoMetadata {
  filename?: string | null;
  width?: number | null;
  height?: number | null;
  fps?: number | null;
  frame_count?: number | null;
  duration_sec?: number | null;
  has_audio?: boolean | null;
  audio_codec?: string | null;
  video_codec?: string | null;
}

/** Per-stage timings reported by video_verifier.step_timings. */
export interface VideoTimings {
  preprocessing_and_extraction_sec?: number | null;
  visual_classification_sec?: number | null;
  claim_research_sec?: number | null;
  total_verification_sec?: number | null;
  pipeline_stage_timings?: Record<string, number>;
}

/** Visual classifier signal, reported separately from the factual verdict. */
export interface VideoClassifierSignal {
  available?: boolean;
  predicted_label?: string | null;
  confidence?: number | null;
  probabilities?: Record<string, number> | null;
  model_name?: string | null;
  model_version?: string | null;
  experimental_warning?: string | null;
  error?: string | null;
}

/** Candidate-claim ranking summary for video. */
export interface VideoCandidateClaims {
  candidate_claims_count?: number | null;
  selected_claims_count?: number | null;
  rejected_claims_count?: number | null;
  all_candidate_claims?: CandidateClaim[];
  verified_claims?: any[];
  research_skipped_reason?: string | null;
}

/** Modality-specific structured payload. Each modality exposes only its own keys. */
export interface ModalitySpecificPayload {
  video?: VideoModalitySpecificPayload;
  image?: ImageModalitySpecificPayload;
  audio?: AudioModalitySpecificPayload;
}

export interface VideoModalitySpecificPayload {
  video_metadata?: VideoMetadata | null;
  frames?: {
    extracted_count?: number | null;
    ocr_processed_count?: number | null;
    ocr_frames_with_text_count?: number | null;
    ocr_sampling_policy?: string | null;
    ocr_observations?: OcrObservation[];
    ocr_warnings?: string[];
  };
  transcript?: {
    text?: string | null;
    segments?: TranscriptionSegment[];
    whisper_segments_count?: number | null;
    language?: string | null;
  };
  fusion?: {
    fused_entries_count?: number | null;
    fused_entries?: FusedEntry[];
    review_flags_count?: number | null;
    deduplication_summary?: Record<string, any>;
  };
  candidate_claims?: VideoCandidateClaims;
  visual_classifier?: VideoClassifierSignal;
  timings?: VideoTimings;
  whisper_recommendations?: Record<string, any>;
  pipeline_warnings?: string[];
  pipeline_errors?: string[];
}

export interface ImageModalitySpecificPayload {
  ocr_results?: OcrResultItem[];
  detected_objects?: DetectedObject[];
  image_claim?: string | null;
  normalized_claim?: Record<string, any>;
  ai_generation?: SyntheticMediaDetails | null;
  ai_generation_note?: string | null;
}

export interface AudioModalitySpecificPayload {
  audio_metadata?: Record<string, any>;
  synthetic_speech_detection?: SyntheticMediaDetails | null;
  transcription?: Record<string, any>;
  claim_extraction?: Record<string, any>;
  consistency_analysis?: Record<string, any> | null;
}

export interface SystemHealthResponse {
  status: string;
  service: string;
  version: string;
  api_keys_configured: {
    groq: boolean;
    tavily: boolean;
  };
  modules: {
    text: { available: boolean; description: string };
    image: { available: boolean; description: string };
    video: { available: boolean; description: string };
    audio: { available: boolean; description: string };
  };
}

export interface ArchitectureNode {
  id: string;
  label: string;
  level: number;
  modality?: ModalityType | 'shared' | 'core';
  description: string;
  entryPoint?: string;
  models?: string[];
  latency?: string;
  limitations?: string;
  children?: string[];
  parentId?: string;
}

export interface EvaluationBenchmark {
  title: string;
  modelName: string;
  architecture: string;
  datasetName: string;
  sampleCount: number;
  primaryMetric: { label: string; value: string };
  secondaryMetrics: { label: string; value: string }[];
  confusionMatrix?: {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
    uncertain?: number;
  };
  notes: string;
  status: 'verified' | 'experimental' | 'pilot';
}
