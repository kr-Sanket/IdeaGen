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
  GapStatus,
  OpenAlexSearchParams,
  OpenAlexSearchResponse,
  OpenAlexIngestResponse,
  ResearchKnowledge,
  OpportunityEvolveRequest,
  OpportunityEvolveResponse,
  WorkspaceStats,
  ProjectPaperSelectionItem,
  PreflightResponse,
  ResearchRun,
  RunLogsResponse,
  ProjectRecord,
  ProjectCreateRequest,
  BackendHealth,
  SystemStatus,
} from '../types';

const API_BASE = '/api/v1';

const PAPERS_PAGE_SIZE = 100;
const PAPERS_MAX_PAGES = 50;

/** Renders a FastAPI error `detail` (a string, or a list of validation errors for 422s). */
function formatErrorDetail(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (item && typeof item === 'object' && 'msg' in item) {
        const loc = Array.isArray((item as { loc?: unknown }).loc)
          ? ((item as { loc: unknown[] }).loc).filter((l) => l !== 'body' && l !== 'query').join('.')
          : '';
        const msg = String((item as { msg: unknown }).msg);
        return loc ? `${loc}: ${msg}` : msg;
      }
      return JSON.stringify(item);
    });
    return parts.join('; ');
  }
  if (detail != null) return JSON.stringify(detail);
  return null;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMsg = `HTTP Error ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      const detail = formatErrorDetail(errJson?.detail);
      if (detail) errorMsg = `${detail} (HTTP ${res.status})`;
    } catch {
      // response has no json body
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

/** fetch() wrapper that turns network failures into a readable message. */
async function request(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch {
    throw new Error('Could not reach the IdeaGen API. Check that the backend server is running.');
  }
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const res = await request(url, {
    method: 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return handleResponse<T>(res);
}

export const api = {
  /**
   * Fetches real workspace overview metrics from the backend, optionally scoped to a project.
   */
  async getWorkspaceStats(projectId?: string): Promise<WorkspaceStats> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    const res = await request(`${API_BASE}/stats${query}`);
    return handleResponse<WorkspaceStats>(res);
  },

  /**
   * Fetches run identity, cutoffs, screening decisions, and corpus counts.
   */
  async getRunMetadata(): Promise<RunMetadata> {
    const res = await request(`${API_BASE}/run-metadata`);
    return handleResponse<RunMetadata>(res);
  },

  /**
   * Lists all opportunities generated in the completed run with lifecycle mappings.
   * Optionally filtered by project_id.
   */
  async getOpportunities(projectId?: string): Promise<OpportunitySummary[]> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    const res = await request(`${API_BASE}/opportunities${query}`);
    return handleResponse<OpportunitySummary[]>(res);
  },

  /**
   * Fetches the 16-section opportunity proposal report.
   */
  async getOpportunityReport(opportunityId: string, projectId?: string): Promise<OpportunityReport> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    const res = await request(`${API_BASE}/opportunities/${opportunityId}/report${query}`);
    return handleResponse<OpportunityReport>(res);
  },

  /**
   * Fetches chronological temporal evidence linked to this opportunity.
   */
  async getOpportunityEvidence(opportunityId: string, projectId?: string): Promise<OpportunityEvidenceItem[]> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    const res = await request(`${API_BASE}/opportunities/${opportunityId}/evidence${query}`);
    return handleResponse<OpportunityEvidenceItem[]>(res);
  },

  /**
   * Fetches recorded lifecycle state transition history.
   */
  async getOpportunityHistory(opportunityId: string, projectId?: string): Promise<OpportunityHistoryItem[]> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    const res = await request(`${API_BASE}/opportunities/${opportunityId}/history${query}`);
    return handleResponse<OpportunityHistoryItem[]>(res);
  },

  /**
   * Lists all corpus papers, optionally scoped by project_id.
   */
  async getPapers(projectId?: string): Promise<PaperItem[]> {
    const all: PaperItem[] = [];
    for (let page = 0; page < PAPERS_MAX_PAGES; page++) {
      const params = new URLSearchParams({
        limit: String(PAPERS_PAGE_SIZE),
        offset: String(page * PAPERS_PAGE_SIZE),
      });
      if (projectId) params.set('project_id', projectId);
      const res = await request(`${API_BASE}/papers?${params.toString()}`);
      const batch = await handleResponse<PaperItem[]>(res);
      all.push(...batch);
      if (batch.length < PAPERS_PAGE_SIZE) break;
    }
    return all;
  },

  /**
   * Fetches a single paper's details.
   */
  async getPaper(paperId: string): Promise<PaperItem> {
    const res = await request(`${API_BASE}/papers/${paperId}`);
    return handleResponse<PaperItem>(res);
  },

  /**
   * Fetches extracted statements, optionally filtered by paper, statement type, or project.
   */
  async getStatements(paperId?: string, type?: string, projectId?: string): Promise<StatementItem[]> {
    if (paperId) {
      const res = await request(`${API_BASE}/papers/${paperId}/statements`);
      return handleResponse<StatementItem[]>(res);
    }
    const params = new URLSearchParams();
    if (type && type !== 'ALL') params.append('type', type);
    if (projectId) params.append('project_id', projectId);
    const res = await request(`${API_BASE}/statements?${params.toString()}`);
    return handleResponse<StatementItem[]>(res);
  },

  /**
   * Retrieves full evidence trace for a statement (paper, section, page, passage snippet).
   */
  async getEvidenceDetail(statementId: string): Promise<EvidenceDetail> {
    const res = await request(`${API_BASE}/evidence/${statementId}`);
    return handleResponse<EvidenceDetail>(res);
  },

  /**
   * Retrieves research gaps, optionally filtered by status and project_id.
   */
  async getGaps(status?: GapStatus, projectId?: string): Promise<GapItem[]> {
    const params = new URLSearchParams();
    if (status) params.set('status', status);
    if (projectId) params.set('project_id', projectId);
    const query = params.toString() ? `?${params.toString()}` : '';
    const res = await request(`${API_BASE}/gaps${query}`);
    return handleResponse<GapItem[]>(res);
  },

  /**
   * Retrieves a specific research gap by ID.
   */
  async getGap(gapId: string): Promise<GapItem> {
    const res = await request(`${API_BASE}/gaps/${encodeURIComponent(gapId)}`);
    return handleResponse<GapItem>(res);
  },

  /**
   * Searches OpenAlex. Does not persist anything.
   */
  async searchOpenAlex({ query, limit, startDate, endDate }: OpenAlexSearchParams): Promise<OpenAlexSearchResponse> {
    const params = new URLSearchParams({ query });
    if (limit !== undefined) params.set('limit', String(limit));
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const res = await request(`${API_BASE}/search/openalex?${params.toString()}`);
    return handleResponse<OpenAlexSearchResponse>(res);
  },

  /**
   * Ingests one OpenAlex work into the corpus (metadata, open-access PDF and passages when available).
   * Post-ingestion evolution only runs when the server enables it.
   * If projectId is provided, the paper is automatically associated with that project in project_papers.
   */
  async ingestOpenAlex(openalexId: string, doi?: string | null, projectId?: string | null): Promise<OpenAlexIngestResponse> {
    return postJson<OpenAlexIngestResponse>(`${API_BASE}/papers/ingest/openalex`, {
      openalex_id: openalexId,
      ...(doi ? { doi } : {}),
      ...(projectId ? { project_id: projectId } : {}),
    });
  },

  /**
   * Associates an existing paper with a research project.
   */
  async associatePaperToProject(projectId: string, paperId: string): Promise<{ status: string; project_id: string; paper_id: string }> {
    return postJson<{ status: string; project_id: string; paper_id: string }>(
      `${API_BASE}/projects/${encodeURIComponent(projectId)}/papers/${encodeURIComponent(paperId)}`,
    );
  },

  /**
   * Runs the LLM paper-understanding pipeline (extracts research statements from stored passages).
   */
  async understandPaper(paperId: string): Promise<ResearchKnowledge> {
    return postJson<ResearchKnowledge>(`${API_BASE}/papers/${encodeURIComponent(paperId)}/understand`);
  },

  /**
   * Runs gap discovery over a project's stored research statements (or the whole corpus when no
   * project is given). Returns the gaps created by this call, assigned to that project.
   */
  async discoverGaps(cutoffDate?: string, projectId?: string): Promise<GapItem[]> {
    const params = new URLSearchParams();
    if (cutoffDate) params.set('cutoff_date', cutoffDate);
    if (projectId) params.set('project_id', projectId);
    const query = params.toString() ? `?${params.toString()}` : '';
    return postJson<GapItem[]>(`${API_BASE}/gaps/discover${query}`);
  },

  /**
   * Generates and persists a research opportunity from a VALIDATED gap.
   * The returned opportunity_id is the generation record id.
   */
  async generateOpportunity(gapId: string, cutoffDate?: string, projectId?: string): Promise<OpportunityReport> {
    const params = new URLSearchParams();
    if (cutoffDate) params.set('cutoff_date', cutoffDate);
    if (projectId) params.set('project_id', projectId);
    const query = params.toString() ? `?${params.toString()}` : '';
    return postJson<OpportunityReport>(`${API_BASE}/gaps/${encodeURIComponent(gapId)}/opportunity${query}`);
  },

  /**
   * Classifies corpus papers against an opportunity and applies lifecycle transitions.
   * Accepts a lifecycle id or a generation record id.
   */
  async evolveOpportunity(opportunityId: string, body: OpportunityEvolveRequest, projectId?: string): Promise<OpportunityEvolveResponse> {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    return postJson<OpportunityEvolveResponse>(
      `${API_BASE}/opportunities/${encodeURIComponent(opportunityId)}/evolve${query}`,
      body,
    );
  },

  /**
   * Verifies backend connectivity truthfully: 'online' (API and database answer), 'degraded'
   * (the API answers but reports a database error) or 'unreachable' (no API response, including a
   * dev-proxy 502 when the frontend cannot connect to the backend).
   */
  async checkBackendHealth(): Promise<BackendHealth> {
    try {
      const res = await fetch('/api/health', { method: 'GET', cache: 'no-store' });
      if (res.ok) return 'online';
      return res.status === 503 ? 'degraded' : 'unreachable';
    } catch {
      return 'unreachable';
    }
  },

  /** Configured LLM / embedding providers with a live Ollama availability check. */
  async getSystemStatus(): Promise<SystemStatus> {
    const res = await request(`${API_BASE}/system/status`, { cache: 'no-store' });
    return handleResponse<SystemStatus>(res);
  },

  // ── Projects (server is the source of truth) ───────────────────────────

  async listProjects(): Promise<ProjectRecord[]> {
    const res = await request(`${API_BASE}/projects`);
    return handleResponse<ProjectRecord[]>(res);
  },

  /** Creates a project; idempotent for an existing ID (the stored project is returned unchanged). */
  async createProject(body: ProjectCreateRequest): Promise<ProjectRecord> {
    return postJson<ProjectRecord>(`${API_BASE}/projects`, body);
  },

  // ── Phase 3: Project Selections, Preflight & Run Execution ─────────────

  /**
   * Retrieves all papers currently selected for a project from the server.
   */
  async getProjectPaperSelections(projectId: string): Promise<ProjectPaperSelectionItem[]> {
    const res = await request(`${API_BASE}/projects/${encodeURIComponent(projectId)}/selections`);
    return handleResponse<ProjectPaperSelectionItem[]>(res);
  },

  /**
   * Adds or updates persistent paper selections for a project.
   */
  async saveProjectPaperSelections(
    projectId: string,
    selections: ProjectPaperSelectionItem[]
  ): Promise<ProjectPaperSelectionItem[]> {
    return postJson<ProjectPaperSelectionItem[]>(
      `${API_BASE}/projects/${encodeURIComponent(projectId)}/selections`,
      { selections }
    );
  },

  /**
   * Removes a single paper selection from a project.
   */
  async removeProjectPaperSelection(projectId: string, openalexId: string): Promise<void> {
    const res = await request(
      `${API_BASE}/projects/${encodeURIComponent(projectId)}/selections/${encodeURIComponent(openalexId)}`,
      { method: 'DELETE' }
    );
    await handleResponse<{ status: string }>(res);
  },

  /**
   * Clears all paper selections for a project.
   */
  async clearProjectPaperSelections(projectId: string): Promise<void> {
    const res = await request(
      `${API_BASE}/projects/${encodeURIComponent(projectId)}/selections`,
      { method: 'DELETE' }
    );
    await handleResponse<{ status: string }>(res);
  },

  /**
   * Runs preflight check on selected papers for a project.
   */
  async getPreflightCheck(projectId: string): Promise<PreflightResponse> {
    const res = await request(`${API_BASE}/projects/${encodeURIComponent(projectId)}/preflight`);
    return handleResponse<PreflightResponse>(res);
  },

  /**
   * Creates and launches an asynchronous research run for a project.
   */
  async createResearchRun(projectId: string, plannedStages?: string[]): Promise<ResearchRun> {
    return postJson<ResearchRun>(`${API_BASE}/projects/${encodeURIComponent(projectId)}/runs`, {
      planned_stages: plannedStages,
    });
  },

  /**
   * Lists all research runs for a project.
   */
  async getProjectRuns(projectId: string): Promise<ResearchRun[]> {
    const res = await request(`${API_BASE}/projects/${encodeURIComponent(projectId)}/runs`);
    return handleResponse<ResearchRun[]>(res);
  },

  /**
   * Retrieves current status and per-paper breakdown of a research run.
   */
  async getRunStatus(runId: string): Promise<ResearchRun> {
    const res = await request(`${API_BASE}/runs/${encodeURIComponent(runId)}`);
    return handleResponse<ResearchRun>(res);
  },

  /**
   * Retries failed or interrupted stages idempotently.
   */
  async retryRun(runId: string): Promise<ResearchRun> {
    return postJson<ResearchRun>(`${API_BASE}/runs/${encodeURIComponent(runId)}/retry`);
  },

  /**
   * Retrieves technical execution logs for a run.
   */
  async getRunLogs(runId: string): Promise<RunLogsResponse> {
    const res = await request(`${API_BASE}/runs/${encodeURIComponent(runId)}/logs`);
    return handleResponse<RunLogsResponse>(res);
  },
};
