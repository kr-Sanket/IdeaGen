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

/** Research gap status (backend/ideagen/discovery/models.py GapStatus). */
export type GapStatus = 'CANDIDATE' | 'VALIDATED' | 'REJECTED';

/** ResearchGapResponseDTO (backend/ideagen/discovery/models.py). */
export interface GapItem {
  id: string;
  gap_statement: string;
  research_problem: string;
  status: GapStatus | string;
  evidence_count: number;
  supporting_statement_ids: string[];
  supporting_paper_ids: string[];
  limitations: string[];
  future_work: string[];
  validation_reasoning?: string | null;
  validation_confidence?: number | null;
  created_at?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface ResearchGapRefineRequest {
  refined_statement: string;
  refined_problem: string;
  refinement_rationale: string;
  why_insufficient?: string;
}

// ── Workflow actions (POST endpoints) ───────────────────────────────────

/** OpenAlexPaperResultDTO (GET /api/v1/search/openalex). */
export interface OpenAlexPaper {
  id: string;
  doi: string | null;
  title: string;
  authors: string[];
  publication_year: number | null;
  publication_date: string | null;
  abstract: string;
  is_oa: boolean | null;
  pdf_url: string | null;
  landing_page_url: string | null;
  citation_count: number | null;
}

/** OpenAlexSearchResponse. */
export interface OpenAlexSearchResponse {
  query: string;
  source: string;
  count: number;
  total_count: number | null;
  papers: OpenAlexPaper[];
}

export interface OpenAlexSearchParams {
  query: string;
  limit?: number;
  startDate?: string;
  endDate?: string;
}

/** IngestionStatus (backend/ideagen/ingestion/openalex_ingestion.py). */
export type IngestionStatus = 'INGESTED' | 'NO_PDF' | 'SKIPPED_DUPLICATE' | 'ERROR';

/** EvolutionHookStatus (backend/ideagen/ingestion/openalex_ingestion.py). */
export type EvolutionHookStatus = 'DISABLED' | 'SUCCESS' | 'PARTIAL_SUCCESS' | 'FAILED' | 'SKIPPED';

/** One per-opportunity outcome inside _format_evolution_result. */
export interface IncrementalEvolutionOutcome {
  opportunity_id: string;
  status: string;
  initial_status: string | null;
  final_status: string | null;
  relationship: string | null;
  state_changed: boolean;
  evidence_id: string | null;
  history_id: string | null;
  skip_reason: string | null;
  error_message: string | null;
}

/** Post-ingestion evolution result (main.py _format_evolution_result); may also be a raw error dict. */
export interface IncrementalEvolutionResult {
  paper_id?: string;
  run_status?: string;
  paper_title?: string | null;
  publication_date?: string | null;
  candidate_count?: number;
  evaluated_count?: number;
  skipped_count?: number;
  grounding_rejected_count?: number;
  failed_count?: number;
  outcomes?: IncrementalEvolutionOutcome[];
  error?: string | null;
  [key: string]: unknown;
}

/** OpenAlexIngestResponse (POST /api/v1/papers/ingest/openalex). */
export interface OpenAlexIngestResponse {
  paper_id: string | null;
  source_id: string;
  status: IngestionStatus | string;
  has_pdf: boolean;
  document_id: string | null;
  passage_count: number;
  error_message: string | null;
  batch_id: string;
  doi: string | null;
  evolution_status: EvolutionHookStatus | string | null;
  evolution_result: IncrementalEvolutionResult | null;
  project_id?: string | null;
  project_associated?: boolean;
  already_in_project?: boolean;
}

/** ResearchKnowledgeResponseDTO (POST /api/v1/papers/{id}/understand). */
export interface ResearchKnowledge {
  id: string;
  paper_id: string;
  research_problem: string | null;
  research_questions: string[];
  methodology: string | null;
  datasets: string[];
  findings: string[];
  limitations: string[];
  future_work: string[];
  statement_count: number;
  model_name: string;
  extraction_schema_version: string;
  extracted_at: string;
  extraction_status: string;
  error_message: string | null;
}

/** OpportunityEvolveRequest. */
export interface OpportunityEvolveRequest {
  paper_ids: string[];
  cutoff_date?: string;
}

/** OpportunityEvolutionEventDTO. */
export interface OpportunityEvolutionEvent {
  paper_id: string;
  relationship: string;
  previous_status: string;
  new_status: string;
  state_changed: boolean;
  evidence_id: string | null;
  history_id: string | null;
}

/** OpportunityEvolveResponse. opportunity_id is always the lifecycle Opportunity id. */
export interface OpportunityEvolveResponse {
  opportunity_id: string;
  initial_status: string;
  final_status: string;
  processed_count: number;
  skipped_count: number;
  failed_count: number;
  events: OpportunityEvolutionEvent[];
  errors: { paper_id: string; reason: string }[];
}

// ── Workspace & Multi-Project Types ─────────────────────────────────────

export interface WorkspaceStats {
  total_papers: number;
  analyzed_papers: number;
  total_passages: number;
  total_statements: number;
  total_gaps: number;
  candidate_gaps: number;
  validated_gaps: number;
  rejected_gaps: number;
  total_opportunities: number;
  validated_opportunities: number;
  analysis_status: string;
}

export type ProjectStatus = 'ACTIVE' | 'ARCHIVED' | 'DRAFT';

/** A research project as stored by the backend (GET/POST /api/v1/projects). */
export interface ProjectRecord {
  id: string;
  title: string;
  topic: string | null;
  description: string | null;
  temporal_cutoff: string | null;
  status: string | null;
  created_at: string | null;
  updated_at: string | null;
  /** Papers in the project's corpus (project_papers). */
  paper_count: number;
  /** Papers queued for the next research run. */
  selection_count: number;
  gap_count: number;
  opportunity_count: number;
  run_count: number;
}

export interface ProjectCreateRequest {
  id: string;
  title: string;
  topic?: string;
  description?: string;
  temporal_cutoff?: string;
}

export interface ProjectDeleteSummary {
  runs_deleted: number;
  run_papers_deleted: number;
  selections_deleted: number;
  paper_links_removed: number;
  gaps_deleted: number;
  gap_evidence_deleted: number;
  research_opportunities_deleted: number;
  prior_art_assessments_deleted: number;
  refinement_iterations_deleted: number;
  lifecycle_opportunities_deleted: number;
  opportunity_evidence_deleted: number;
  opportunity_history_deleted: number;
}

export interface ProjectDeleteResponse {
  project_id: string;
  deleted: boolean;
  deleted_at: string;
  summary: ProjectDeleteSummary;
}

export type BackendHealth = 'online' | 'degraded' | 'unreachable';

/** GET /api/v1/system/status: configured providers and live availability (no secrets). */
export interface SystemStatus {
  database: { connected: boolean; dialect: string | null };
  llm: { provider: string; model: string | null; base_url: string | null; api_key_configured?: boolean };
  embedding: { provider: string; model: string | null; dimension: number | null; base_url: string | null };
  ollama: {
    reachable: boolean;
    installed_models: string[];
    missing_models: string[];
    error: string | null;
    auto_manage?: boolean;
    is_owned?: boolean;
    active_leases?: number;
    idle_timeout?: number;
    process_running?: boolean;
  } | null;
  openalex: { email_configured: boolean; proxy: string | null };
}

export interface ResearchProject {
  id: string;
  name: string;
  topic: string;
  description: string;
  createdAt: string;
  updatedAt: string;
  status: ProjectStatus;
  temporalCutoff?: string;
  isDefaultBenchmark?: boolean;
}

export interface ProjectActivityItem {
  id: string;
  timestamp: string;
  type: 'EVOLUTION' | 'INGESTION' | 'UNDERSTANDING' | 'GAP_DISCOVERY' | 'OPPORTUNITY_GENERATION';
  title: string;
  description: string;
  targetId?: string;
  badgeText?: string;
  badgeVariant?: 'default' | 'success' | 'warning' | 'info';
}

export interface ProjectAttentionRun {
  runId: string;
  title: string;
  status: 'AWAITING_REVIEW' | 'FAILED' | 'INTERRUPTED' | 'READY';
  description: string;
  actionLabel: string;
  actionRoute: string;
}

// ── Phase 3 Research Workflow & Execution Types ─────────────────────────

export interface ProjectPaperSelectionItem {
  id?: string;
  project_id?: string;
  openalex_id: string;
  paper_id?: string;
  doi?: string | null;
  title: string;
  authors?: string[];
  publication_year?: number | null;
  abstract?: string | null;
  pdf_url?: string | null;
  landing_page_url?: string | null;
  has_pdf?: boolean;
  selected_at?: string;
}

export type PreflightPaperStatus = 'NEW' | 'CORPUS_MATCH' | 'NO_PDF' | 'METADATA_ONLY';

export interface PreflightPaperItem {
  openalex_id: string;
  paper_id?: string;
  doi?: string | null;
  title: string;
  authors: string[];
  publication_year?: number | null;
  has_pdf: boolean;
  pdf_url?: string | null;
  status: PreflightPaperStatus;
  corpus_paper_id?: string | null;
  warning?: string | null;
  is_pre_cutoff?: boolean;
}

export interface PreflightStageItem {
  stage_id: string;
  stage?: string;
  name: string;
  description: string;
  estimated_action?: string;
}

/** Papers already in the project corpus; a run processes them in addition to queued selections. */
export interface PreflightCorpus {
  total: number;
  with_passages: number;
  statements_extracted: number;
  pending_extraction: number;
  metadata_only: number;
}

export interface PreflightResponse {
  project_id: string;
  corpus?: PreflightCorpus | null;
  /** True when the run has something to process (queued selections or a non-empty corpus). */
  runnable?: boolean;
  total_selected: number;
  new_papers: number;
  corpus_matches: number;
  pdf_available: number;
  metadata_only: number;
  papers: PreflightPaperItem[];
  planned_stages: PreflightStageItem[];
  warnings: string[];
  summary?: {
    total_selected: number;
    new_papers: number;
    corpus_matches: number;
    has_pdf: number;
    metadata_only: number;
  };
}

export type RunStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'PARTIAL_SUCCESS' | 'FAILED' | 'AWAITING_REVIEW' | 'INTERRUPTED';

export interface RunPaper {
  id: string;
  openalex_id: string;
  paper_id?: string;
  doi?: string | null;
  title: string;
  status: string;
  corpus_paper_id?: string | null;
  has_pdf: boolean;
  error_message?: string | null;
  passages_count: number;
  statements_count: number;
}

export interface RunStage {
  stage_id: string;
  stage?: string;
  name: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'PARTIAL_SUCCESS' | 'FAILED' | 'SKIPPED' | 'INTERRUPTED';
  message?: string | null;
  error?: string | null;
}

export interface RunSummary {
  total_papers: number;
  completed_papers: number;
  successful_papers: number;
  failed_papers: number;
  failed_stages?: string[];
  total_statements: number;
  next_actions: string[];
  completed_at?: string;
}

export interface ResearchRun {
  id: string;
  project_id: string;
  status: RunStatus;
  current_stage?: string | null;
  stage_progress?: any;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  error_message?: string | null;
  summary?: any;
  papers: RunPaper[];
  stages: RunStage[];
}

export interface RunLogEntry {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  stage: string;
  message: string;
}

export interface RunLogsResponse {
  run_id: string;
  status: string;
  logs: RunLogEntry[];
}
