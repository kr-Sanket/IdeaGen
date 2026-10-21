import React, { useCallback, useEffect, useState, useMemo } from 'react';
import {
  BookOpen,
  FileText,
  Layers,
  Search,
  RefreshCw,
  Bookmark
} from 'lucide-react';
import type { PaperItem, StatementItem, TemporalRole } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { EmptyState } from '../common/EmptyState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getTemporalRoleBadgeVariant, getTemporalRoleLabel, getStatementTypeBadgeVariant } from '../common/badgeVariants';
import { NotRecorded, RecordedValue } from '../common/NotRecorded';

interface CorpusInspectorViewProps {
  initialPaperId?: string | null;
  onInspectEvidence: (statementId: string) => void;
}

type RoleFilter = 'ALL' | TemporalRole;

const ROLE_FILTER_ORDER: readonly TemporalRole[] = ['BASELINE', 'LATER_EVALUATION', 'DATED', 'UNDATED'];

function roleFilterLabel(role: RoleFilter): string {
  switch (role) {
    case 'ALL':
      return 'All Papers';
    case 'BASELINE':
      return 'Baseline';
    case 'LATER_EVALUATION':
      return 'Later Evaluation';
    case 'DATED':
      return 'Dated (no cutoff)';
    case 'UNDATED':
      return 'Undated';
  }
}

export const CorpusInspectorView: React.FC<CorpusInspectorViewProps> = ({
  initialPaperId,
  onInspectEvidence,
}) => {
  const [papers, setPapers] = useState<PaperItem[]>([]);
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(initialPaperId || null);
  const [statements, setStatements] = useState<StatementItem[]>([]);

  const [loadingPapers, setLoadingPapers] = useState<boolean>(true);
  const [loadingStatements, setLoadingStatements] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [statementsError, setStatementsError] = useState<string | null>(null);
  const [statementsReloadKey, setStatementsReloadKey] = useState<number>(0);

  // Filters
  const [paperRoleFilter, setPaperRoleFilter] = useState<RoleFilter>('ALL');
  const [statementTypeFilter, setStatementTypeFilter] = useState<string>('ALL');
  const [statementSearch, setStatementSearch] = useState<string>('');

  // Papers load once on mount and on explicit refresh; selection changes do not refetch.
  const fetchPapers = useCallback(() => {
    setLoadingPapers(true);
    setError(null);
    api.getPapers()
      .then((data) => {
        setPapers(data);
        setSelectedPaperId((prev) => prev ?? (data.length > 0 ? data[0].id : null));
      })
      .catch((err: Error) => setError(err.message || 'Failed to load corpus papers'))
      .finally(() => setLoadingPapers(false));
  }, []);

  useEffect(() => {
    fetchPapers();
  }, [fetchPapers]);

  // Honour deep-link changes to the selected paper (adjust state during render, no effect).
  const [prevInitialPaperId, setPrevInitialPaperId] = useState<string | null | undefined>(initialPaperId);
  if (initialPaperId !== prevInitialPaperId) {
    setPrevInitialPaperId(initialPaperId);
    if (initialPaperId) setSelectedPaperId(initialPaperId);
  }

  // Fetch statements only for an explicitly selected paper.
  useEffect(() => {
    if (!selectedPaperId) return;
    let cancelled = false;
    setLoadingStatements(true);
    setStatementsError(null);
    api.getStatements(selectedPaperId)
      .then((data) => {
        if (!cancelled) setStatements(data);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setStatements([]);
          setStatementsError(err.message || 'Failed to load statements for the selected paper.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingStatements(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedPaperId, statementsReloadKey]);

  const roleFilters = useMemo<RoleFilter[]>(() => {
    const present = new Set(papers.map((p) => p.temporal_role).filter(Boolean));
    return ['ALL', ...ROLE_FILTER_ORDER.filter((r) => present.has(r))];
  }, [papers]);

  const filteredPapers = useMemo(() => {
    if (paperRoleFilter === 'ALL') return papers;
    return papers.filter((p) => p.temporal_role === paperRoleFilter);
  }, [papers, paperRoleFilter]);

  const selectedPaper = useMemo(() => {
    return papers.find((p) => p.id === selectedPaperId) || null;
  }, [papers, selectedPaperId]);

  // Statement type filter options are derived from the loaded data.
  const statementTypes = useMemo(() => {
    const types = new Set(statements.map((s) => s.statement_type.toUpperCase()));
    return ['ALL', ...Array.from(types).sort()];
  }, [statements]);

  const filteredStatements = useMemo(() => {
    return statements.filter((s) => {
      if (statementTypeFilter !== 'ALL') {
        if (s.statement_type.toUpperCase() !== statementTypeFilter) return false;
      }
      if (statementSearch.trim()) {
        const q = statementSearch.toLowerCase();
        const textMatch = (s.statement_text || '').toLowerCase().includes(q);
        const passMatch = (s.passage_text || '').toLowerCase().includes(q);
        const secMatch = (s.section || '').toLowerCase().includes(q);
        return textMatch || passMatch || secMatch;
      }
      return true;
    });
  }, [statements, statementTypeFilter, statementSearch]);

  const handleSelectPaper = (paperId: string) => {
    if (paperId === selectedPaperId) return;
    setStatementTypeFilter('ALL');
    setSelectedPaperId(paperId);
  };

  if (loadingPapers) {
    return <LoadingState message="Loading corpus papers..." minHeight="400px" />;
  }

  if (error) {
    return (
      <div className="u-stack-lg">
        <ErrorBanner
          title="Could Not Load Corpus"
          message={error}
          onRetry={fetchPapers}
        />
      </div>
    );
  }

  return (
    <div className="view-container animate-fade-in">
      {/* Header */}
      <div className="view-header">
        <div>
          <div className="u-row">
            <h1 className="view-title">Corpus & Statement Provenance Inspector</h1>
            <span className="app-badge badge-neutral">{papers.length} Papers</span>
          </div>
          <p className="view-subtitle">
            Inspect corpus papers, their temporal role relative to the configured cutoff, and extracted statements with linked source passages.
          </p>
        </div>
        <button
          onClick={fetchPapers}
          className="btn btn-secondary"
          title="Refresh papers"
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Main 2-Column Split: Paper List & Statements Pane */}
      <div className="corpus-layout">
        {/* Left Column: Papers List & Role Filter */}
        <div className="corpus-papers-pane">
          <div className="card">
            <div className="card-header">
              <div className="u-row">
                <BookOpen size={16} className="tone-indigo" />
                <h3 className="card-title">Corpus Papers</h3>
              </div>
              <span className="card-header-meta">
                {filteredPapers.length} / {papers.length}
              </span>
            </div>

            {/* Filter Pills */}
            <div className="pane-toolbar">
              {roleFilters.map((role) => (
                <button
                  key={role}
                  onClick={() => setPaperRoleFilter(role)}
                  className={`filter-pill-sm ${paperRoleFilter === role ? 'active' : ''}`}
                  aria-pressed={paperRoleFilter === role}
                >
                  {roleFilterLabel(role)}
                </button>
              ))}
            </div>

            {/* Paper Selection List */}
            <div className="corpus-paper-list">
              {filteredPapers.length === 0 && (
                <p className="empty-note card-body">No papers match this filter.</p>
              )}
              {filteredPapers.map((paper) => {
                const isSelected = paper.id === selectedPaperId;

                return (
                  <button
                    type="button"
                    key={paper.id}
                    onClick={() => handleSelectPaper(paper.id)}
                    className={`corpus-paper-item ${isSelected ? 'selected' : ''}`}
                    aria-current={isSelected ? 'true' : undefined}
                  >
                    <span className="u-row u-between corpus-paper-item-head">
                      <Badge
                        label={getTemporalRoleLabel(paper.temporal_role)}
                        variant={getTemporalRoleBadgeVariant(paper.temporal_role)}
                      />
                      <span className="corpus-paper-date">
                        <RecordedValue value={paper.publication_date} missingLabel="No date" />
                      </span>
                    </span>

                    <span className="corpus-paper-item-title u-block">
                      {paper.title || paper.source_id}
                    </span>

                    <span className="corpus-paper-item-meta">
                      <span>Source: {paper.source_id}</span>
                      <span>·</span>
                      <span>{paper.passage_count} passages</span>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Selected Paper Details & Extracted Statements */}
        <div className="corpus-detail-pane">
          {selectedPaper ? (
            <div className="u-stack-lg">
              {/* Paper Summary Card */}
              <div className="card">
                <div className="card-header">
                  <div className="u-row">
                    <FileText size={16} className="tone-cyan" />
                    <h3 className="card-title">Selected Paper Metadata</h3>
                  </div>
                  <Badge
                    label={getTemporalRoleLabel(selectedPaper.temporal_role)}
                    variant={getTemporalRoleBadgeVariant(selectedPaper.temporal_role)}
                  />
                </div>
                <div className="card-body u-stack">
                  <h2 className="paper-detail-title">
                    {selectedPaper.title || selectedPaper.source_id}
                  </h2>

                  <div className="u-row u-wrap u-row-loose u-text-sm u-text-secondary u-mono">
                    <span>Publication Date: <RecordedValue value={selectedPaper.publication_date} /></span>
                    <span>Source ID: {selectedPaper.source_id}</span>
                    <span>Passages: {selectedPaper.passage_count}</span>
                    {selectedPaper.doi && <span>DOI: {selectedPaper.doi}</span>}
                  </div>

                  {selectedPaper.abstract && (
                    <div className="section-divider">
                      <span className="label-caps">Abstract Excerpt</span>
                      <p className="abstract-scroll">
                        {selectedPaper.abstract}
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Extracted Statements Section */}
              <div className="card">
                <div className="card-header">
                  <div className="u-row">
                    <Layers size={16} className="tone-indigo" />
                    <h3 className="card-title">Extracted Statements & Linked Passages</h3>
                  </div>
                  <span className="card-header-meta">
                    {filteredStatements.length} Statements
                  </span>
                </div>

                {/* Filter & Search Bar */}
                <div className="pane-toolbar u-between">
                  <div className="search-input-wrapper-sm">
                    <Search className="search-icon-sm" />
                    <input
                      type="text"
                      value={statementSearch}
                      onChange={(e) => setStatementSearch(e.target.value)}
                      placeholder="Filter statements text or section..."
                      className="search-input-sm"
                      aria-label="Filter statements"
                    />
                  </div>

                  <div className="u-row u-row-tight u-scroll-x">
                    {statementTypes.map((stType) => (
                      <button
                        key={stType}
                        onClick={() => setStatementTypeFilter(stType)}
                        className={`filter-pill-sm ${statementTypeFilter === stType ? 'active' : ''}`}
                        aria-pressed={statementTypeFilter === stType}
                      >
                        {stType === 'ALL' ? 'All Types' : stType}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="card-body">
                  {statementsError ? (
                    <ErrorBanner
                      title="Could Not Load Statements"
                      message={statementsError}
                      onRetry={() => setStatementsReloadKey((k) => k + 1)}
                    />
                  ) : loadingStatements ? (
                    <LoadingState message="Fetching statements for selected paper..." minHeight="250px" />
                  ) : filteredStatements.length === 0 ? (
                    <EmptyState
                      title="No Statements Found"
                      description={statementSearch ? 'No statements match your search query.' : 'No statements extracted for the selected filter.'}
                    />
                  ) : (
                    <div className="u-stack-md">
                      {filteredStatements.map((stmt) => (
                        <div key={stmt.id} className="statement-inspector-card">
                          <div className="statement-inspector-header">
                            <div className="u-row">
                              <Badge
                                label={stmt.statement_type}
                                variant={getStatementTypeBadgeVariant(stmt.statement_type)}
                              />
                              <Badge label="Model-Extracted" variant="model-generated" />
                            </div>

                            <button
                              onClick={() => onInspectEvidence(stmt.id)}
                              className="btn btn-secondary btn-sm"
                              title="Inspect linked source passage"
                            >
                              <Bookmark size={14} className="tone-indigo" />
                              <span>Inspect Source</span>
                            </button>
                          </div>

                          {/* Statement Text */}
                          <div className="statement-inspector-text">
                            "{stmt.statement_text}"
                          </div>

                          {/* Provenance Location Details */}
                          <div className="statement-inspector-meta">
                            {stmt.section && <span>Section: {stmt.section}</span>}
                            {stmt.page_start !== null && stmt.page_start !== undefined && (
                              <span>Page: {stmt.page_start}{stmt.page_end && stmt.page_end !== stmt.page_start ? `–${stmt.page_end}` : ''}</span>
                            )}
                            {stmt.confidence !== null && stmt.confidence !== undefined && (
                              <span>
                                Model confidence: {(stmt.confidence * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>

                          {/* Linked passage preview if populated */}
                          {stmt.passage_text ? (
                            <div className="passage-preview-box">
                              <div className="u-row u-row-tight passage-preview-header">
                                <FileText size={14} />
                                <span>Linked source passage</span>
                              </div>
                              <p className="passage-preview-text">
                                {stmt.passage_text}
                              </p>
                            </div>
                          ) : (
                            <span className="u-text-sm">
                              <NotRecorded label="No linked source passage" />
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <EmptyState
              title="No Paper Selected"
              description="Select a paper from the list on the left to inspect its extracted statements and source passages."
            />
          )}
        </div>
      </div>
    </div>
  );
};
