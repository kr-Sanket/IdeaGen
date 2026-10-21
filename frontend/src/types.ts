/**
 * IdeaGen Read-Only Explorer TypeScript Data Contracts.
 *
 * Aligned with the FastAPI backend contract:
 *   - backend/ideagen/api/schema.py (DTOs)
 *   - backend/ideagen/api/main.py (resolve_paper_temporal_role, run-metadata assembly)
 *   - backend/ideagen/discovery/screening.py (ScreeningReport.to_dict / PaperScreeningResult.to_dict)
 *   - experiments/runner.py (compute_configuration_fingerprint)
 *   - backend/ideagen/opportunity/models.py (lifecycle OpportunityStatus, OpportunityRelationship)
 *   - backend/ideagen/generation/models.py (generation OpportunityStatus)
 */

// ── Enumerations mirrored from backend enums ────────────────────────────

/** Lifecycle states (backend/ideagen/opportunity/models.py OpportunityStatus). */
export type LifecycleStatus = 'OPEN' | 'STRENGTHENED' | 'PARTIALLY_ADDRESSED' | 'RESOLVED';

export const LIFECYCLE_STATUSES: readonly LifecycleStatus[] = [
  'OPEN',
  'STRENGTHENED',
  'PARTIALLY_ADDRESSED',
  'RESOLVED',
];

/** Evidence relationship types (backend/ideagen/opportunity/models.py OpportunityRelationship). */
export type EvidenceRelationship = 'SUPPORTS' | 'ADDRESSES' | 'CONTRADICTS' | 'EXTENDS' | 'UNRELATED';

/** Generation pipeline status (backend/ideagen/generation/models.py OpportunityStatus). */
export type GenerationStatus =
  | 'DRAFT'
  | 'VALIDATED'
  | 'REFINED'
  | 'REJECTED_INSUFFICIENTLY_DIFFERENTIATED'
  | 'INSUFFICIENTLY_DIFFERENTIATED';

/** Values produced by resolve_paper_temporal_role in backend/ideagen/api/main.py. */
export type TemporalRole = 'BASELINE' | 'LATER_EVALUATION' | 'UNDATED' | 'DATED';

// ── Run metadata ────────────────────────────────────────────────────────

export interface RunMetadataCounts {
  papers: number;
  baseline_papers: number;
  later_evaluation_papers: number;
  statements: number;
  research_gaps: number;
  opportunities: number;
  lifecycle_opportunities: number;
  evidence_records: number;
  history_records: number;
}

/** Top passage snippet attached to a screened paper (PassageSnippet.to_dict). */
export interface ScreeningPassageSnippet {
  passage_id: string | null;
  passage_index: number | null;
  section: string | null;
  page_start: number | null;
  page_end: number | null;
  similarity: number;
  snippet: string;
}

/** Decision values documented on PaperScreeningResult.decision. */
export type ScreeningDecisionValue =
  | 'SELECTED_ABOVE_THRESHOLD'
  | 'SELECTED_BELOW_THRESHOLD_FLOOR'
  | 'EXCLUDED_BELOW_THRESHOLD'
  | 'EXCLUDED_CAPACITY_LIMIT';

/** One ranked paper entry (PaperScreeningResult.to_dict). */
export interface ScreeningRankedPaper {
  paper_id: string;
  filename: string;
  title: string | null;
  publication_date: string | null;
  score: number;
  rank: number;
  decision: ScreeningDecisionValue | string;
  reason: string;
  paper_level_similarity: number;
  body_similarity: number;
  floor_included: boolean;
  is_pre_cutoff: boolean;
  temporal_workflow_note: string | null;
  top_passages: ScreeningPassageSnippet[];
}

/** Stored relevance screening report (ScreeningReport.to_dict). */
export interface RelevanceScreeningReport {
  enabled: boolean;
  topic: string;
  threshold: number;
  min_papers: number;
  max_papers: number | null;
  total_eligible_papers: number;
  selected_count: number;
  excluded_count: number;
  floor_warning: string | null;
  metric_interpretation_note: string;
  ranked_papers: ScreeningRankedPaper[];
  input_signature?: Record<string, unknown>;
}

/** Nested passage-selection bounds in the configuration fingerprint. */
export interface FingerprintPassageSelection {
  max_passages?: number;
  max_total_words?: number;
  max_abstract_chars?: number;
  [key: string]: unknown;
}

/** Nested Stage 3 gap discovery parameters in the configuration fingerprint. */
export interface FingerprintGapDiscovery {
  similarity_threshold?: number;
  min_supporting_evidence?: number;
  [key: string]: unknown;
}

/** Nested relevance screening configuration in the configuration fingerprint. */
export interface FingerprintRelevanceScreening {
  enabled?: boolean;
  topic?: string | null;
  max_papers?: number | null;
  min_papers?: number | null;
  threshold?: number | null;
  [key: string]: unknown;
}

/** Configuration fingerprint (experiments/runner.py compute_configuration_fingerprint). */
export interface ConfigurationFingerprint {
  llm_provider?: string;
  llm_model?: string;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_dimension?: number;
  ollama_timeout?: number;
  passage_selection?: FingerprintPassageSelection | null;
  extraction_schema_version?: string;
  gap_discovery?: FingerprintGapDiscovery | null;
  max_refinement_iterations?: number;
  temporal_cutoff?: string | null;
  relevance_screening?: FingerprintRelevanceScreening | null;
  enable_stage_5?: boolean;
  stage_5_batch_window_days?: number | null;
  stage_5_cutoffs?: string[] | null;
  stage_5_chunk_size?: number | null;
  stage_5_max_retries?: number | null;
  stage_5_passage_selection?: FingerprintPassageSelection | null;
  [key: string]: unknown;
}

/** RunMetadataDTO. */
export interface RunMetadata {
  run_fingerprint: ConfigurationFingerprint | null;
  baseline_cutoff: string | null;
  evaluation_cutoffs: string[] | null;
  stage_5_enabled: boolean;
  stage_5_batch_window_days: number | null;
  relevance_screening: RelevanceScreeningReport | null;
  discovery_completion: Record<string, unknown> | null;
  total_papers: number;
  total_statements: number;
  total_gaps: number;
  total_opportunities: number;
  counts: RunMetadataCounts | null;
  database_status: string;
  metric_interpretation_note: string | null;
}

// ── Opportunities ───────────────────────────────────────────────────────

/** OpportunityListItemDTO. */
export interface OpportunitySummary {
  id: string;
  gap_id: string;
  research_problem: string;
  identified_gap: string;
  proposed_research_direction: string;
  status: GenerationStatus | string;
  confidence: number;
  created_at: string | null;
  supporting_statement_count: number;
  supporting_paper_count: number;
  lifecycle_opportunity_id: string | null;
  lifecycle_status: LifecycleStatus | null;
  lifecycle_confidence: number | null;
  evidence_count: number;
}

/** EvidenceDetailDTO. */
export interface EvidenceDetail {
  statement_id: string;
  statement_text: string;
  statement_type: string;
  paper_id: string;
  paper_title: string;
  passage_id?: string | null;
  section?: string | null;
  page_start?: number | null;
  page_end?: number | null;
  passage_text?: string | null;
}

/** FeasibilityDTO. */
// Model-generated; null means the stored opportunity has no value (not measured).
export interface FeasibilityAssessment {
  required_data: string | null;
  required_compute: string | null;
  implementation_complexity: string | null;
  experimental_setup: string | null;
  feasibility_score: number | null;
  summary?: string | null;
}

/** NoveltyDTO. */
export interface NoveltyAssessment {
  assessment: string | null;
  reasoning: string | null;
  differentiation: string | null;
  confidence: number | null;
  overlapping_statement_ids?: string[];
  overlapping_paper_ids?: string[];
}

/** OpportunityReportDTO. */
export interface OpportunityReport {
  opportunity_id: string;
  gap_id: string;
  research_problem: string;
  evidence_of_problem: EvidenceDetail[];
  existing_approaches: string[];
  identified_gap: string;
  why_existing_work_is_insufficient: string | null;
  proposed_research_direction: string;
  reasoning: string;
  technical_approach: string;
  expected_contribution: string;
  feasibility: FeasibilityAssessment;
  prior_art_assessment: Record<string, unknown>;
  differentiation: string | null;
  supporting_papers: PaperItem[];
  supporting_evidence: EvidenceDetail[];
  confidence: number;
  novelty_assessment: NoveltyAssessment;
  status: GenerationStatus | string;
  created_at: string;
  generation_model: string;
  validation_model: string;
  embedding_model: string;
  refinement_iteration_count: number;
  pipeline_version: string;
  lifecycle_status?: LifecycleStatus | null;
  lifecycle_opportunity_id?: string | null;
}

/** OpportunityEvidenceDTO. */
export interface OpportunityEvidenceItem {
  id: string;
  opportunity_id: string;
  paper_id: string;
  paper_title: string | null;
  paper_publication_date: string | null;
  batch_id: string | null;
  evaluation_cutoff: string | null;
  relationship: EvidenceRelationship;
  confidence: number | null;
  supporting_text: string | null;
  /** Computed server-side by is_evidence_grounded_for_paper. */
  verified: boolean | null;
  created_at: string | null;
}

/** OpportunityHistoryDTO. */
export interface OpportunityHistoryItem {
  id: string;
  opportunity_id: string;
  previous_status: LifecycleStatus;
  new_status: LifecycleStatus;
  trigger_type: EvidenceRelationship;
  evidence_id: string | null;
  created_at: string | null;
}

// ── Corpus ──────────────────────────────────────────────────────────────

/** PaperResponseDTO. */
export interface PaperItem {
  id: string;
  source_type: string;
  source_id: string;
  title: string | null;
  publication_date: string | null;
  doi: string | null;
  abstract: string | null;
  created_at?: string | null;
  document_count: number;
  passage_count: number;
  knowledge_status: string;
  content_hash?: string | null;
  temporal_role?: TemporalRole | null;
  is_baseline?: boolean | null;
}

/** StatementResponseDTO. */
export interface StatementItem {
  id: string;
  paper_id: string;
  paper_title: string;
  passage_id?: string | null;
  statement_text: string;
  statement_type: string;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  confidence?: number | null;
  created_at?: string | null;
  passage_text?: string | null;
}

export interface GapItem {
  id: string;
  gap_statement: string;
  research_problem: string;
  status: string;
  evidence_count: number;
  supporting_statement_ids: string[];
  supporting_paper_ids: string[];
  limitations: string[];
  future_work: string[];
  validation_reasoning?: string | null;
  validation_confidence?: number | null;
  created_at?: string | null;
}
