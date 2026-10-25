import React, { useCallback, useRef, useState, useEffect, useMemo } from 'react';
import {
  Search,
  ExternalLink,
  Plus,
  Loader2,
  FileText,
  BookOpen,
  ArrowRight,
  Quote,
  Calendar,
  Users,
  CheckSquare,
  Square,
  Trash2,
  Play,
  X,
} from 'lucide-react';
import type {
  OpenAlexPaper,
  OpenAlexSearchResponse,
  OpenAlexIngestResponse,
  ResearchKnowledge,
  ResearchProject,
  ProjectPaperSelectionItem,
  PaperItem,
} from '../../types';
import { api } from '../../api/client';
import { useProjectContext } from '../../projects/projectStore';
import { EmptyState } from '../common/EmptyState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { IngestEvolutionSummary } from '../workflow/IngestEvolutionSummary';
import { UnderstandResult } from '../workflow/UnderstandResult';
import { RunPreflightModal } from '../workflow/RunPreflightModal';
import { RunProgressModal } from '../workflow/RunProgressModal';

const RESULT_LIMITS = [10, 25, 50] as const;

type IngestState =
  | { phase: 'pending' }
  | { phase: 'done'; result: OpenAlexIngestResponse }
  | { phase: 'error'; message: string };

type UnderstandState =
  | { phase: 'pending' }
  | { phase: 'done'; result: ResearchKnowledge }
  | { phase: 'error'; message: string };

interface SearchPapersViewProps {
  onNavigate: (route: string) => void;
  project?: ResearchProject;
}

interface StoredSearchContext {
  query: string;
  startDate: string;
  endDate: string;
  limit: number;
  response: OpenAlexSearchResponse | null;
  ingestStates: Record<string, IngestState>;
}

function getSearchStorageKey(projectId?: string | null): string {
  return `ideagen_search_state_${projectId || 'default'}`;
}

function loadStoredSearchContext(projectId?: string | null): StoredSearchContext | null {
  try {
    const raw = sessionStorage.getItem(getSearchStorageKey(projectId));
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveStoredSearchContext(projectId: string | null | undefined, ctx: StoredSearchContext): void {
  try {
    sessionStorage.setItem(getSearchStorageKey(projectId), JSON.stringify(ctx));
  } catch (err) {
    console.warn('Failed to save search state to sessionStorage:', err);
  }
}

function formatAuthors(authors: string[]): string | null {
  if (authors.length === 0) return null;
  if (authors.length <= 3) return authors.join(', ');
  return `${authors.slice(0, 3).join(', ')} et al.`;
}

function ingestStatusExplanation(result: OpenAlexIngestResponse): string {
  const where = result.project_associated ? "this project's corpus" : 'the shared literature store (not linked to a project)';
  switch (result.status) {
    case 'INGESTED':
      return `Added to ${where} with full text: ${result.passage_count} passage${result.passage_count === 1 ? '' : 's'} stored.`;
    case 'NO_PDF':
      return `Added to ${where} as metadata only: no open-access PDF could be downloaded and parsed, so no passages were stored and statements cannot be extracted from this paper.`;
    case 'SKIPPED_DUPLICATE':
      if (result.already_in_project) {
        return "This paper is already present in this project's corpus; nothing new was stored.";
      }
      if (result.project_associated) {
        return "Paper was already present in the shared literature store and has now been associated with this project.";
      }
      return 'This paper already exists in the global corpus; nothing new was stored.';
    case 'ERROR':
      return result.error_message || 'Ingestion reported an error; see the message below.';
    default:
      return `Ingestion returned status ${result.status}.`;
  }
}

function getIngestBadge(result: OpenAlexIngestResponse): { label: string; variant: 'status-positive' | 'status-warning' | 'status-negative' | 'neutral' } {
  if (result.status === 'INGESTED') {
    return result.project_associated
      ? { label: 'Added to Project (Full Text)', variant: 'status-positive' }
      : { label: 'Stored, Not Linked to Project', variant: 'status-warning' };
  }
  if (result.status === 'NO_PDF') {
    return result.project_associated
      ? { label: 'Added to Project (Metadata Only)', variant: 'status-warning' }
      : { label: 'Stored (Metadata Only), Not Linked', variant: 'status-warning' };
  }
  if (result.status === 'SKIPPED_DUPLICATE') {
    if (result.already_in_project) {
      return { label: 'Already in Project Corpus', variant: 'neutral' };
    }
    if (result.project_associated) {
      return { label: 'Added to Project (Existing Paper)', variant: 'status-positive' };
    }
    return { label: 'Duplicate in Global Corpus', variant: 'status-warning' };
  }
  if (result.status === 'ERROR') {
    return { label: 'Ingestion Failed', variant: 'status-negative' };
  }
  return { label: `Status: ${result.status}`, variant: 'neutral' };
}

export const SearchPapersView: React.FC<SearchPapersViewProps> = ({ onNavigate, project }) => {
  const { activeProject } = useProjectContext();
  const currentProject = project || activeProject;

  const [query, setQuery] = useState<string>('');
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [limit, setLimit] = useState<number>(10);

  const [searching, setSearching] = useState<boolean>(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [response, setResponse] = useState<OpenAlexSearchResponse | null>(null);

  // Persistent paper selections (Keyed by paper_id / OpenAlex ID)
  const [selectedPapers, setSelectedPapers] = useState<Record<string, ProjectPaperSelectionItem>>({});

  // Workflow run modals
  const [isPreflightOpen, setIsPreflightOpen] = useState<boolean>(false);
  const [isProgressOpen, setIsProgressOpen] = useState<boolean>(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  // Keyed by OpenAlex work id / paper id. Refs guard against double submission before re-render.
  const [ingestStates, setIngestStates] = useState<Record<string, IngestState>>({});
  const [understandStates, setUnderstandStates] = useState<Record<string, UnderstandState>>({});
  const ingestInFlight = useRef<Set<string>>(new Set());
  const understandInFlight = useRef<Set<string>>(new Set());
  const searchSeq = useRef<number>(0);

  // Which project the in-memory search state belongs to. Persisting is skipped until the state for
  // the current project has been loaded; otherwise the initial empty state (or the previous
  // project's results, during a project switch) would overwrite the saved entry.
  const [stateProjectId, setStateProjectId] = useState<string | null>(null);

  // Load persistent search state on project mount / switch
  useEffect(() => {
    const saved = loadStoredSearchContext(currentProject?.id);
    if (saved) {
      setQuery(saved.query || '');
      setStartDate(saved.startDate || '');
      setEndDate(saved.endDate || '');
      setLimit(saved.limit || 10);
      setResponse(saved.response || null);
      setIngestStates(saved.ingestStates || {});
    } else {
      setQuery('');
      setStartDate('');
      setEndDate('');
      setLimit(10);
      setResponse(null);
      setIngestStates({});
    }
    setSearchError(null);
    setStateProjectId(currentProject?.id ?? null);
  }, [currentProject?.id]);

  // Persist search state whenever it changes
  useEffect(() => {
    if (!currentProject?.id || stateProjectId !== currentProject.id) return;
    saveStoredSearchContext(currentProject.id, {
      query,
      startDate,
      endDate,
      limit,
      response,
      ingestStates,
    });
  }, [currentProject?.id, stateProjectId, query, startDate, endDate, limit, response, ingestStates]);

  // Fetch project corpus papers to identify already-associated papers
  const [projectCorpusPapers, setProjectCorpusPapers] = useState<PaperItem[]>([]);

  const refreshProjectCorpus = useCallback(() => {
    if (!currentProject?.id) return;
    api
      .getPapers(currentProject.id)
      .then((papers) => setProjectCorpusPapers(papers))
      .catch((err) => console.warn('Failed to fetch project corpus papers:', err));
  }, [currentProject?.id]);

  useEffect(() => {
    refreshProjectCorpus();
  }, [refreshProjectCorpus]);

  const projectDoiSet = useMemo(() => {
    const set = new Set<string>();
    for (const p of projectCorpusPapers) {
      if (p.doi) set.add(p.doi.toLowerCase().trim());
    }
    return set;
  }, [projectCorpusPapers]);

  const projectSourceIdSet = useMemo(() => {
    const set = new Set<string>();
    for (const p of projectCorpusPapers) {
      if (p.source_id) set.add(p.source_id.toLowerCase().trim());
      if (p.id) set.add(p.id.toLowerCase().trim());
    }
    return set;
  }, [projectCorpusPapers]);

  const isInProjectCorpus = useCallback(
    (paper: OpenAlexPaper): boolean => {
      if (paper.doi && projectDoiSet.has(paper.doi.toLowerCase().trim())) return true;
      if (paper.id && projectSourceIdSet.has(paper.id.toLowerCase().trim())) return true;
      const normOaId = paper.id.replace(/^https?:\/\/openalex\.org\//i, '').toLowerCase();
      if (normOaId && projectSourceIdSet.has(normOaId)) return true;
      return false;
    },
    [projectDoiSet, projectSourceIdSet],
  );

  const handleClearSearch = () => {
    setQuery('');
    setResponse(null);
    setSearchError(null);
    if (currentProject?.id) {
      sessionStorage.removeItem(getSearchStorageKey(currentProject.id));
    }
  };

  // Load persistent paper selections for current project
  useEffect(() => {
    if (!currentProject?.id) return;
    api
      .getProjectPaperSelections(currentProject.id)
      .then((items) => {
        const map: Record<string, ProjectPaperSelectionItem> = {};
        for (const it of items) {
          const key = it.openalex_id || it.paper_id || '';
          if (key) map[key] = it;
        }
        setSelectedPapers(map);
      })
      .catch((err) => {
        console.warn('Failed to fetch project selections from server:', err);
      });
  }, [currentProject?.id]);

  const toggleSelection = (paper: OpenAlexPaper) => {
    if (!currentProject?.id) return;
    const isSelected = Boolean(selectedPapers[paper.id]);

    if (isSelected) {
      setSelectedPapers((prev) => {
        const next = { ...prev };
        delete next[paper.id];
        return next;
      });
      api.removeProjectPaperSelection(currentProject.id, paper.id).catch((err) => {
        console.error('Failed to remove paper selection from server:', err);
      });
    } else {
      const item: ProjectPaperSelectionItem = {
        openalex_id: paper.id,
        paper_id: paper.id,
        doi: paper.doi || undefined,
        title: paper.title,
        publication_year: paper.publication_year || undefined,
        authors: paper.authors || [],
        has_pdf: Boolean(paper.pdf_url),
        pdf_url: paper.pdf_url || undefined,
        landing_page_url: paper.landing_page_url || undefined,
      };
      setSelectedPapers((prev) => ({ ...prev, [paper.id]: item }));
      api.saveProjectPaperSelections(currentProject.id, [item]).catch((err) => {
        console.error('Failed to save paper selection to server:', err);
      });
    }
  };

  const handleSelectAllInResponse = () => {
    if (!response || !currentProject?.id) return;
    const newItems: ProjectPaperSelectionItem[] = [];
    const updatedMap = { ...selectedPapers };

    for (const paper of response.papers) {
      if (!updatedMap[paper.id]) {
        const item: ProjectPaperSelectionItem = {
          openalex_id: paper.id,
          paper_id: paper.id,
          doi: paper.doi || undefined,
          title: paper.title,
          publication_year: paper.publication_year || undefined,
          authors: paper.authors || [],
          has_pdf: Boolean(paper.pdf_url),
          pdf_url: paper.pdf_url || undefined,
          landing_page_url: paper.landing_page_url || undefined,
        };
        updatedMap[paper.id] = item;
        newItems.push(item);
      }
    }

    setSelectedPapers(updatedMap);
    if (newItems.length > 0) {
      api.saveProjectPaperSelections(currentProject.id, newItems).catch((err) => {
        console.error('Failed to batch save selections:', err);
      });
    }
  };

  const handleClearSelections = async () => {
    if (!currentProject?.id) return;
    setSelectedPapers({});
    try {
      await api.clearProjectPaperSelections(currentProject.id);
    } catch (err) {
      console.error('Failed to clear selections on server:', err);
    }
  };

  const executeSearch = useCallback(() => {
    const trimmed = query.trim();
    if (!trimmed || searching) return;
    const seq = ++searchSeq.current;
    setSearching(true);
    setSearchError(null);
    api
      .searchOpenAlex({
        query: trimmed,
        limit,
        startDate: startDate || undefined,
        endDate: endDate || undefined,
      })
      .then((data) => {
        if (seq === searchSeq.current) setResponse(data);
      })
      .catch((err: Error) => {
        if (seq === searchSeq.current) {
          setResponse(null);
          setSearchError(err.message || 'OpenAlex search failed.');
        }
      })
      .finally(() => {
        if (seq === searchSeq.current) setSearching(false);
      });
  }, [query, limit, startDate, endDate, searching]);

  const runSearch = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      executeSearch();
    },
    [executeSearch],
  );

  const ingest = (paper: OpenAlexPaper) => {
    if (ingestInFlight.current.has(paper.id)) return;
    ingestInFlight.current.add(paper.id);
    setIngestStates((prev) => ({ ...prev, [paper.id]: { phase: 'pending' } }));
    api
      .ingestOpenAlex(paper.id, paper.doi, currentProject?.id)
      .then((result) => {
        setIngestStates((prev) => ({ ...prev, [paper.id]: { phase: 'done', result } }));
        refreshProjectCorpus();
      })
      .catch((err: Error) =>
        setIngestStates((prev) => ({
          ...prev,
          [paper.id]: { phase: 'error', message: err.message || 'Ingestion failed.' },
        })),
      )
      .finally(() => ingestInFlight.current.delete(paper.id));
  };

  const understand = (paperId: string) => {
    if (understandInFlight.current.has(paperId)) return;
    understandInFlight.current.add(paperId);
    setUnderstandStates((prev) => ({ ...prev, [paperId]: { phase: 'pending' } }));
    api
      .understandPaper(paperId)
      .then((result) =>
        setUnderstandStates((prev) => ({ ...prev, [paperId]: { phase: 'done', result } })),
      )
      .catch((err: Error) =>
        setUnderstandStates((prev) => ({
          ...prev,
          [paperId]: { phase: 'error', message: err.message || 'Statement extraction failed.' },
        })),
      )
      .finally(() => understandInFlight.current.delete(paperId));
  };

  const dateRangeInvalid = Boolean(startDate && endDate && startDate > endDate);
  const selectedCount = Object.keys(selectedPapers).length;

  return (
    <div className="view-container animate-fade-in">
      <div className="view-header">
        <div>
          <h1 className="view-title">Search &amp; Select Literature</h1>
          <p className="view-subtitle">
            Search OpenAlex for domain literature, select candidate papers into the project queue,
            and launch the end-to-end evidence discovery pipeline.
          </p>
        </div>

        {selectedCount > 0 && (
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setIsPreflightOpen(true)}
            id="btn-header-preflight"
          >
            <Play size={14} fill="currentColor" />
            <span>Review &amp; Start Run ({selectedCount})</span>
          </button>
        )}
      </div>

      <ol className="workflow-steps" aria-label="Research workflow">
        <li className="workflow-step is-current">
          <span className="workflow-step-num">1</span>
          <span>Search &amp; select papers</span>
        </li>
        <li className="workflow-step">
          <button
            type="button"
            className="workflow-step-link"
            onClick={() => setIsPreflightOpen(true)}
            disabled={selectedCount === 0}
          >
            <span className="workflow-step-num">2</span>
            <span>Preflight review &amp; run</span>
          </button>
        </li>
        <li className="workflow-step">
          <button
            type="button"
            className="workflow-step-link"
            onClick={() => onNavigate(currentProject ? `#/projects/${currentProject.id}/gaps` : '#/gaps')}
          >
            <span className="workflow-step-num">3</span>
            <span>Discover gaps &amp; generate</span>
          </button>
        </li>
        <li className="workflow-step">
          <button type="button" className="workflow-step-link" onClick={() => onNavigate('#/opportunities')}>
            <span className="workflow-step-num">4</span>
            <span>Track &amp; evolve opportunities</span>
          </button>
        </li>
      </ol>

      <form className="card search-form-card" onSubmit={runSearch} role="search">
        <div className="card-body u-stack">
          <div className="search-form-row">
            <div className="search-input-wrapper search-input-grow">
              <Search className="search-icon" />
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Topic, title, keywords, OpenAlex ID or DOI"
                className="search-input"
                aria-label="Search query"
                required
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={searching || !query.trim() || dateRangeInvalid}
            >
              {searching ? <Loader2 size={14} className="spin" /> : <Search size={14} />}
              <span>{searching ? 'Searching…' : 'Search OpenAlex'}</span>
            </button>
            {Boolean(query || response) && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleClearSearch}
                title="Clear current search query and results"
              >
                <X size={14} />
                <span>Clear</span>
              </button>
            )}
          </div>
          <div className="search-form-row search-form-filters">
            <label className="form-field">
              <span className="form-label">Published from</span>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="form-input"
              />
            </label>
            <label className="form-field">
              <span className="form-label">Published to</span>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="form-input"
              />
            </label>
            <label className="form-field">
              <span className="form-label">Results</span>
              <select
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                className="form-input"
              >
                {RESULT_LIMITS.map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
          </div>
          {dateRangeInvalid && (
            <p className="form-error" role="alert">The start date must be on or before the end date.</p>
          )}
        </div>
      </form>

      {searchError && (
        <ErrorBanner
          title="Search Failed"
          message={searchError}
          onRetry={executeSearch}
        />
      )}

      {searching && !response && (
        <div className="inline-pending" role="status">
          <Loader2 size={16} className="spin" />
          <span>Searching OpenAlex…</span>
        </div>
      )}

      {!response && !searching && !searchError && (
        <EmptyState
          title="Start with a literature search"
          description="Results come directly from OpenAlex. Select multiple papers to form your research project corpus, or add individual papers directly."
          icon={<BookOpen size={24} />}
        />
      )}

      {response && (
        <div className="u-stack" aria-busy={searching}>
          <div className="u-row u-between u-wrap" style={{ alignItems: 'center' }}>
            <p className="u-text-sm u-text-secondary">
              Showing {response.count} result{response.count === 1 ? '' : 's'} for{' '}
              <span className="u-text-strong">"{response.query}"</span>
              {typeof response.total_count === 'number' && (
                <> · {response.total_count.toLocaleString()} matching works in OpenAlex</>
              )}
            </p>

            {response.papers.length > 0 && (
              <div className="u-row" style={{ gap: '0.5rem' }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-xs"
                  onClick={handleSelectAllInResponse}
                >
                  <CheckSquare size={13} />
                  <span>Select all on page</span>
                </button>
              </div>
            )}
          </div>

          {response.papers.length === 0 ? (
            <EmptyState
              title="No papers found"
              description="OpenAlex returned no works for this query and date range. Try broader keywords or remove the date filter."
            />
          ) : (
            <div className="search-results-list">
              {response.papers.map((paper) => {
                const isSelected = Boolean(selectedPapers[paper.id]);
                const inProjectCorpus = isInProjectCorpus(paper);
                const ingestState = ingestStates[paper.id];
                const authors = formatAuthors(paper.authors);
                const ingestedPaperId =
                  ingestState?.phase === 'done' ? ingestState.result.paper_id : null;
                const understandState = ingestedPaperId ? understandStates[ingestedPaperId] : undefined;
                const canUnderstand =
                  ingestState?.phase === 'done' &&
                  ingestState.result.status === 'INGESTED' &&
                  ingestState.result.passage_count > 0 &&
                  Boolean(ingestedPaperId);

                return (
                  <article
                    key={paper.id}
                    className={`card search-result-card ${isSelected ? 'is-selected-card' : ''}`}
                    style={
                      isSelected
                        ? {
                            borderColor: 'var(--primary)',
                            boxShadow: '0 0 0 1px rgba(99, 102, 241, 0.4)',
                          }
                        : undefined
                    }
                  >
                    <div className="card-body u-stack">
                      <div className="u-row u-between u-wrap search-result-head">
                        <div className="u-row u-wrap" style={{ alignItems: 'center', gap: '0.625rem' }}>
                          {/* Selection Checkbox Pill */}
                          <button
                            type="button"
                            className={`btn ${isSelected ? 'btn-primary' : 'btn-secondary'} btn-xs`}
                            onClick={() => toggleSelection(paper)}
                            id={`select-paper-${paper.id}`}
                            aria-pressed={isSelected}
                            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem' }}
                          >
                            {isSelected ? <CheckSquare size={14} /> : <Square size={14} />}
                            <span>{isSelected ? 'Selected' : 'Select for Run'}</span>
                          </button>

                          {paper.pdf_url ? (
                            <Badge label="Open-access PDF" variant="status-positive" />
                          ) : (
                            <Badge
                              label="No PDF listed"
                              variant="neutral"
                              title="Metadata only; full text passages cannot be extracted"
                            />
                          )}

                          {typeof paper.citation_count === 'number' && (
                            <span className="meta-stat-pill">
                              <Quote size={12} />
                              <span>{paper.citation_count.toLocaleString()} citations</span>
                            </span>
                          )}
                        </div>

                        <span className="u-mono u-text-xs u-text-tertiary">{paper.id}</span>
                      </div>

                      <h3 className="search-result-title">{paper.title}</h3>

                      <div className="u-row u-wrap u-row-loose u-text-sm u-text-secondary">
                        {authors && (
                          <span className="u-row u-row-tight">
                            <Users size={14} />
                            <span>{authors}</span>
                          </span>
                        )}
                        <span className="u-row u-row-tight">
                          <Calendar size={14} />
                          <span>{paper.publication_date || paper.publication_year || 'Date not recorded'}</span>
                        </span>
                        {paper.doi && <span className="u-mono u-text-xs">DOI: {paper.doi}</span>}
                      </div>

                      {paper.abstract ? (
                        <p className="prose-small clamp-3">{paper.abstract}</p>
                      ) : (
                        <p className="empty-note">OpenAlex has no abstract for this work.</p>
                      )}

                      <div className="u-row u-between u-wrap search-result-actions">
                        <div className="u-row u-wrap">
                          {paper.landing_page_url && (
                            <a
                              href={paper.landing_page_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="btn-link"
                            >
                              <ExternalLink size={12} />
                              <span>Publisher page</span>
                            </a>
                          )}
                          {paper.pdf_url && (
                            <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer" className="btn-link">
                              <FileText size={12} />
                              <span>PDF</span>
                            </a>
                          )}
                        </div>

                        <div className="u-row u-wrap" style={{ gap: '0.5rem', alignItems: 'center' }}>
                          {inProjectCorpus && ingestState?.phase !== 'done' && (
                            <Badge label="In Project Corpus" variant="neutral" />
                          )}

                          {ingestState?.phase !== 'done' && (
                            <button
                              type="button"
                              className="btn btn-secondary btn-sm"
                              onClick={() => ingest(paper)}
                              disabled={ingestState?.phase === 'pending' || inProjectCorpus}
                              aria-label={`Add "${paper.title}" to project corpus`}
                            >
                              {ingestState?.phase === 'pending' ? (
                                <Loader2 size={14} className="spin" />
                              ) : inProjectCorpus ? (
                                <CheckSquare size={14} />
                              ) : (
                                <Plus size={14} />
                              )}
                              <span>
                                {ingestState?.phase === 'pending'
                                  ? 'Adding to Project…'
                                  : inProjectCorpus
                                    ? 'In Project Corpus'
                                    : ingestState?.phase === 'error'
                                      ? 'Retry adding'
                                      : 'Add to Project Corpus'}
                              </span>
                            </button>
                          )}
                        </div>
                      </div>

                      {ingestState?.phase === 'error' && (
                        <div className="action-result is-error" role="alert">
                          <p className="action-result-title">Could not add this paper</p>
                          <p className="u-text-sm">{ingestState.message}</p>
                        </div>
                      )}

                      {ingestState?.phase === 'done' && (
                        <div
                          className={`action-result ${ingestState.result.status === 'ERROR' ? 'is-error' : (ingestState.result.status === 'INGESTED' || ingestState.result.project_associated) ? 'is-success' : 'is-warning'}`}
                          role="status"
                        >
                          <div className="u-row u-wrap" style={{ alignItems: 'center', gap: '0.5rem' }}>
                            {(() => {
                              const badgeInfo = getIngestBadge(ingestState.result);
                              return <Badge label={badgeInfo.label} variant={badgeInfo.variant} />;
                            })()}
                            {ingestState.result.paper_id && (
                              <span className="u-mono u-text-xs u-text-tertiary">
                                Paper ID: {ingestState.result.paper_id}
                              </span>
                            )}
                          </div>
                          <p className="u-text-sm">{ingestStatusExplanation(ingestState.result)}</p>
                          {ingestState.result.error_message && (
                            <p className="u-text-xs u-text-secondary u-mono action-result-detail">
                              {ingestState.result.error_message}
                            </p>
                          )}

                          <IngestEvolutionSummary
                            status={ingestState.result.evolution_status}
                            result={ingestState.result.evolution_result}
                          />

                          <div className="u-row u-wrap action-result-actions">
                            {canUnderstand && ingestedPaperId && understandState?.phase !== 'done' && (
                              <button
                                type="button"
                                className="btn btn-primary btn-sm"
                                onClick={() => understand(ingestedPaperId)}
                                disabled={understandState?.phase === 'pending'}
                              >
                                {understandState?.phase === 'pending' ? (
                                  <Loader2 size={14} className="spin" />
                                ) : (
                                  <FileText size={14} />
                                )}
                                <span>
                                  {understandState?.phase === 'pending'
                                    ? 'Extracting statements… (LLM)'
                                    : understandState?.phase === 'error'
                                      ? 'Retry statement extraction'
                                      : 'Extract statements'}
                                </span>
                              </button>
                            )}
                            {ingestedPaperId && (
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => onNavigate(`#/papers/${ingestedPaperId}`)}
                              >
                                <BookOpen size={14} />
                                <span>Open in corpus</span>
                              </button>
                            )}
                          </div>

                          {understandState?.phase === 'error' && (
                            <div className="action-result is-error" role="alert">
                              <p className="action-result-title">Statement extraction failed</p>
                              <p className="u-text-sm">{understandState.message}</p>
                            </div>
                          )}
                          {understandState?.phase === 'done' && (
                            <UnderstandResult
                              knowledge={understandState.result}
                              action={
                                <button
                                  type="button"
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => onNavigate('#/gaps')}
                                >
                                  <span>Go to research gaps</span>
                                  <ArrowRight size={14} />
                                </button>
                              }
                            />
                          )}
                        </div>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Sticky Selection Action Bar */}
      {selectedCount > 0 && (
        <div className="sticky-selection-tray">
          <div className="u-row" style={{ gap: '0.875rem', alignItems: 'center' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'var(--primary)',
                color: '#fff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 700,
                fontSize: '0.875rem',
                boxShadow: '0 0 10px rgba(99, 102, 241, 0.5)',
              }}
            >
              {selectedCount}
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: '0.9375rem', color: 'var(--text-primary)' }}>
                {selectedCount} Paper{selectedCount === 1 ? '' : 's'} Selected for Research Run
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                Project: <strong style={{ color: 'var(--text-secondary)' }}>{currentProject?.name}</strong> · Persisted to server database
              </div>
            </div>
          </div>

          <div className="u-row" style={{ gap: '0.75rem' }}>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleClearSelections}
              title="Clear all selected papers for this project"
            >
              <Trash2 size={13} />
              <span>Clear Selections</span>
            </button>

            <button
              type="button"
              className="btn btn-primary"
              onClick={() => setIsPreflightOpen(true)}
              id="btn-sticky-start-run"
              style={{ boxShadow: '0 0 15px rgba(99, 102, 241, 0.45)' }}
            >
              <Play size={15} fill="currentColor" />
              <span>Review Preflight &amp; Start Run</span>
            </button>
          </div>
        </div>
      )}

      {/* Preflight Review Modal */}
      {currentProject && (
        <RunPreflightModal
          isOpen={isPreflightOpen}
          onClose={() => setIsPreflightOpen(false)}
          projectId={currentProject.id}
          projectName={currentProject.name}
          onStartRun={(runId) => {
            setActiveRunId(runId);
            setIsProgressOpen(true);
          }}
        />
      )}

      {/* Progress & Live Execution Modal */}
      {currentProject && (
        <RunProgressModal
          isOpen={isProgressOpen}
          onClose={() => setIsProgressOpen(false)}
          runId={activeRunId}
          projectId={currentProject.id}
          onNavigate={onNavigate}
        />
      )}
    </div>
  );
};
