import type {
  RunMetadata,
  OpportunitySummary,
  OpportunityReport,
  OpportunityEvidenceItem,
  OpportunityHistoryItem,
  PaperItem,
  StatementItem,
  EvidenceDetail,
  GapItem,
} from '../types';

const API_BASE = '/api/v1';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMsg = `HTTP Error ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // response has no json body
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

export const api = {
  /**
   * Fetches run identity, cutoffs, screening decisions, and corpus counts.
   */
  async getRunMetadata(): Promise<RunMetadata> {
    const res = await fetch(`${API_BASE}/run-metadata`);
    return handleResponse<RunMetadata>(res);
  },

  /**
   * Lists all opportunities generated in the completed run with lifecycle mappings.
   */
  async getOpportunities(): Promise<OpportunitySummary[]> {
    const res = await fetch(`${API_BASE}/opportunities`);
    return handleResponse<OpportunitySummary[]>(res);
  },

  /**
   * Fetches the 16-section opportunity proposal report.
   */
  async getOpportunityReport(opportunityId: string): Promise<OpportunityReport> {
    const res = await fetch(`${API_BASE}/opportunities/${opportunityId}/report`);
    return handleResponse<OpportunityReport>(res);
  },

  /**
   * Fetches chronological temporal evidence linked to this opportunity.
   */
  async getOpportunityEvidence(opportunityId: string): Promise<OpportunityEvidenceItem[]> {
    const res = await fetch(`${API_BASE}/opportunities/${opportunityId}/evidence`);
    return handleResponse<OpportunityEvidenceItem[]>(res);
  },

  /**
   * Fetches recorded lifecycle state transition history.
   */
  async getOpportunityHistory(opportunityId: string): Promise<OpportunityHistoryItem[]> {
    const res = await fetch(`${API_BASE}/opportunities/${opportunityId}/history`);
    return handleResponse<OpportunityHistoryItem[]>(res);
  },

  /**
   * Lists all corpus papers with baseline vs later-evaluation classifications.
   */
  async getPapers(): Promise<PaperItem[]> {
    const res = await fetch(`${API_BASE}/papers`);
    return handleResponse<PaperItem[]>(res);
  },

  /**
   * Fetches a single paper's details.
   */
  async getPaper(paperId: string): Promise<PaperItem> {
    const res = await fetch(`${API_BASE}/papers/${paperId}`);
    return handleResponse<PaperItem>(res);
  },

  /**
   * Fetches extracted statements, optionally filtered by paper or statement type.
   */
  async getStatements(paperId?: string, type?: string): Promise<StatementItem[]> {
    if (paperId) {
      const res = await fetch(`${API_BASE}/papers/${paperId}/statements`);
      return handleResponse<StatementItem[]>(res);
    }
    const params = new URLSearchParams();
    if (type && type !== 'ALL') params.append('type', type);
    const res = await fetch(`${API_BASE}/statements?${params.toString()}`);
    return handleResponse<StatementItem[]>(res);
  },

  /**
   * Retrieves full evidence trace for a statement (paper, section, page, passage snippet).
   */
  async getEvidenceDetail(statementId: string): Promise<EvidenceDetail> {
    const res = await fetch(`${API_BASE}/evidence/${statementId}`);
    return handleResponse<EvidenceDetail>(res);
  },

  /**
   * Retrieves research gaps.
   */
  async getGaps(): Promise<GapItem[]> {
    const res = await fetch('/api/gaps');
    return handleResponse<GapItem[]>(res);
  },

  /**
   * Retrieves a specific research gap by ID.
   */
  async getGap(gapId: string): Promise<GapItem> {
    const res = await fetch(`/api/gaps/${gapId}`);
    return handleResponse<GapItem>(res);
  },

  /**
   * Verifies backend connectivity truthfully.
   */
  async checkBackendConnection(): Promise<boolean> {
    try {
      const res = await fetch('/api/health', { method: 'GET' });
      return res.ok;
    } catch {
      return false;
    }
  },
};
