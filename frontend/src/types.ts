export interface WorkspaceStats {
  total_papers: number;
  processed_papers: number;
  total_statements: number;
  total_gaps: number;
  candidate_gaps: number;
  validated_gaps: number;
  rejected_gaps: number;
  total_opportunities: number;
  analysis_status: 'IDLE' | 'PROCESSING' | 'COMPLETED' | 'ERROR';
}

export interface PaperResponse {
  id: string;
  source_type: string;
  source_id: string;
  doi?: string | null;
  title?: string | null;
  publication_date?: string | null;
  batch_id: string;
  abstract: string;
  authors: string[];
  passages_count: number;
  statements_count: number;
  has_knowledge: boolean;
  sections: string[];
  created_at?: string;
}

export interface StatementResponse {
  id: string;
  knowledge_id: string;
  paper_id: string;
  passage_id?: string | null;
  statement_text: string;
  statement_type: string;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
  confidence?: number | null;
  paper_title?: string | null;
}

export interface EvidenceDetail {
  statement_id: string;
  statement_text: string;
  statement_type: string;
  confidence?: number | null;
  paper_id: string;
  paper_title?: string | null;
  passage_id?: string | null;
  passage_text?: string | null;
  passage_index?: number | null;
  page_start?: number | null;
  page_end?: number | null;
  section?: string | null;
}

export interface ResearchKnowledgeResponse {
  id: string;
  paper_id: string;
  research_problem?: string | null;
  research_questions: string[];
  methodology?: string | null;
  datasets: string[];
  findings: string[];
  limitations: string[];
  future_work: string[];
  extraction_status: string;
  extracted_at: string;
  statements: StatementResponse[];
}

export interface GapEvidenceItem {
  id: string;
  statement_id: string;
  paper_id: string;
  evidence_role: string;
  statement?: StatementResponse | null;
}

export interface GapResponse {
  id: string;
  gap_statement: string;
  research_problem: string;
  status: 'CANDIDATE' | 'VALIDATED' | 'REJECTED';
  evidence_count: number;
  limitations: string[];
  future_work: string[];
  validation_reasoning?: string | null;
  validation_confidence?: number | null;
  supporting_papers: string[];
  created_at?: string;
}

export interface NoveltyAssessment {
  novelty_level: string;
  reasoning: string;
  novelty_statement: string;
  prior_art_overlap: string[];
  differentiating_factors: string[];
}

export interface FeasibilityAssessment {
  feasibility_score: number;
  technical_barriers: string[];
  required_resources: string[];
  estimated_timeline: string;
}

export interface OpportunityResponse {
  id: string;
  gap_id: string;
  research_problem: string;
  identified_gap: string;
  proposed_research_direction: string;
  reasoning: string;
  technical_approach: string;
  expected_contribution: string;
  feasibility: FeasibilityAssessment;
  novelty_assessment: NoveltyAssessment;
  confidence: number;
  supporting_statement_ids: string[];
  supporting_paper_ids: string[];
  prior_art_statement_ids: string[];
  status: string;
  created_at?: string;
}

export interface PaperProvenance {
  paper_id: string;
  title: string;
  doi?: string | null;
  publication_date?: string | null;
  authors: string[];
}

export interface OpportunityReport {
  opportunity_id: string;
  gap_id: string;
  research_problem: string;
  evidence_of_problem: EvidenceDetail[];
  existing_approaches: EvidenceDetail[];
  identified_gap: string;
  why_existing_work_is_insufficient: string;
  proposed_research_direction: string;
  reasoning: string;
  technical_approach: string;
  expected_contribution: string;
  feasibility: FeasibilityAssessment;
  prior_art_assessment: NoveltyAssessment;
  differentiation: string[];
  supporting_papers: PaperProvenance[];
  supporting_evidence: EvidenceDetail[];
  confidence: number;
  validation_status: string;
  cautious_novelty_statement: string;
  generated_at: string;
}
