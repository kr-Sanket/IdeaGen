import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Sparkles,
  RefreshCw,
  Loader2,
  Wand2,
  Radar,
  Bookmark,
  ArrowRight,
  Info,
} from 'lucide-react';
import type { GapItem, OpportunityReport, OpportunitySummary, PaperItem, ResearchProject } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { EmptyState } from '../common/EmptyState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getGapStatusBadgeVariant, getGenerationStatusBadgeVariant } from '../common/badgeVariants';
import { RecordedValue } from '../common/NotRecorded';
import { displayPercent } from '../../format';
import { GenerationResult } from '../workflow/GenerationResult';
import { formatElapsed, useElapsedSeconds } from '../../hooks/useElapsedSeconds';

type GapFilter = 'ALL' | 'VALIDATED' | 'CANDIDATE' | 'REJECTED';
const GAP_FILTERS: readonly GapFilter[] = ['ALL', 'VALIDATED', 'CANDIDATE', 'REJECTED'];

type GenerationState =
  | { phase: 'done'; report: OpportunityReport }
  | { phase: 'error'; message: string };

type DiscoveryState =
  | { phase: 'done'; created: GapItem[] }
  | { phase: 'error'; message: string };

interface ResearchGapsViewProps {
  project?: ResearchProject | null;
  initialGapId?: string | null;
  onNavigate: (route: string) => void;
  onInspectEvidence: (statementId: string) => void;
}

function gapFilterLabel(filter: GapFilter): string {
  if (filter === 'ALL') return 'All gaps';
  return filter.charAt(0) + filter.slice(1).toLowerCase();
}

export const ResearchGapsView: React.FC<ResearchGapsViewProps> = ({
  project,
  initialGapId,
  onNavigate,
  onInspectEvidence,
}) => {
  const [gaps, setGaps] = useState<GapItem[]>([]);
  const [opportunities, setOpportunities] = useState<OpportunitySummary[]>([]);
  const [papers, setPapers] = useState<PaperItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [opportunitiesError, setOpportunitiesError] = useState<string | null>(null);

  const [filter, setFilter] = useState<GapFilter>('ALL');
  const [selectedGapId, setSelectedGapId] = useState<string | null>(initialGapId ?? null);

  // Discovery (one run at a time).
  const [discoveryCutoff, setDiscoveryCutoff] = useState<string>('');
  const [discoveryStartedAt, setDiscoveryStartedAt] = useState<number | null>(null);
  const [discoveryState, setDiscoveryState] = useState<DiscoveryState | null>(null);
  const discoveryInFlight = useRef<boolean>(false);

  // Generation (one run at a time across all gaps; results kept per gap).
  const [generationCutoff, setGenerationCutoff] = useState<string>('');
  const [generatingGapId, setGeneratingGapId] = useState<string | null>(null);
  const [generationStartedAt, setGenerationStartedAt] = useState<number | null>(null);
  const [generationResults, setGenerationResults] = useState<Record<string, GenerationState>>({});
  const generationInFlight = useRef<boolean>(false);

  const discoveryElapsed = useElapsedSeconds(discoveryStartedAt);
  const generationElapsed = useElapsedSeconds(generationStartedAt);

  const loadOpportunities = useCallback(() => {
    return api
      .getOpportunities(project?.id)
      .then((data) => {
        setOpportunities(data);
        setOpportunitiesError(null);
      })
      .catch((err: Error) => setOpportunitiesError(err.message || 'Could not load existing opportunities.'));
  }, [project?.id]);

  const loadAll = useCallback(
    (showSpinner: boolean) => {
      if (showSpinner) setLoading(true);
      setError(null);
      const gapsPromise = api
        .getGaps(undefined, project?.id)
        .then((data) => {
          setGaps(data);
          setSelectedGapId((prev) => {
            if (prev && data.some((g) => g.id === prev)) return prev;
            const firstValidated = data.find((g) => g.status === 'VALIDATED');
            return firstValidated?.id ?? data[0]?.id ?? null;
          });
        })
        .catch((err: Error) => setError(err.message || 'Failed to load research gaps.'));
      // Paper titles are optional context; a failure here does not block the view.
      api.getPapers(project?.id).then(setPapers).catch(() => setPapers([]));
      Promise.all([gapsPromise, loadOpportunities()]).finally(() => setLoading(false));
    },
    [loadOpportunities, project?.id],
  );

  useEffect(() => {
    loadAll(true);
  }, [loadAll]);

  // Honour deep links (#/gaps/{id}) without an effect.
  const [prevInitialGapId, setPrevInitialGapId] = useState<string | null | undefined>(initialGapId);
  if (initialGapId !== prevInitialGapId) {
    setPrevInitialGapId(initialGapId);
    if (initialGapId) setSelectedGapId(initialGapId);
  }

  const counts = useMemo(() => {
    const c: Record<GapFilter, number> = { ALL: gaps.length, VALIDATED: 0, CANDIDATE: 0, REJECTED: 0 };
    for (const g of gaps) {
      if (g.status === 'VALIDATED' || g.status === 'CANDIDATE' || g.status === 'REJECTED') c[g.status] += 1;
    }
    return c;
  }, [gaps]);

  const filteredGaps = useMemo(
    () => (filter === 'ALL' ? gaps : gaps.filter((g) => g.status === filter)),
    [gaps, filter],
  );

  const opportunitiesByGap = useMemo(() => {
    const map = new Map<string, OpportunitySummary[]>();
    for (const o of opportunities) {
      const list = map.get(o.gap_id) ?? [];
      list.push(o);
      map.set(o.gap_id, list);
    }
    return map;
  }, [opportunities]);

  const paperTitles = useMemo(() => new Map(papers.map((p) => [p.id, p.title || p.source_id])), [papers]);

  const selectedGap = useMemo(() => gaps.find((g) => g.id === selectedGapId) ?? null, [gaps, selectedGapId]);

  const runDiscovery = () => {
    if (discoveryInFlight.current) return;
    const confirmed = window.confirm(
      'Run gap discovery over all stored research statements?\n\n' +
        'This calls the LLM synchronously and may take several minutes. ' +
        'Each run adds newly discovered gaps; existing gaps are kept.',
    );
    if (!confirmed) return;
    discoveryInFlight.current = true;
    setDiscoveryState(null);
    setDiscoveryStartedAt(Date.now());
    api
      .discoverGaps(discoveryCutoff || undefined, project?.id)
      .then((created) => {
        setDiscoveryState({ phase: 'done', created });
        loadAll(false);
      })
      .catch((err: Error) => setDiscoveryState({ phase: 'error', message: err.message || 'Gap discovery failed.' }))
      .finally(() => {
        discoveryInFlight.current = false;
        setDiscoveryStartedAt(null);
      });
  };

  const generate = (gap: GapItem) => {
    if (generationInFlight.current || gap.status !== 'VALIDATED') return;
    const existing = opportunitiesByGap.get(gap.id)?.length ?? 0;
    if (
      existing > 0 &&
      !window.confirm(
        `${existing} opportunit${existing === 1 ? 'y has' : 'ies have'} already been generated from this gap. ` +
          'Generating again creates an additional, separate record. Continue?',
      )
    ) {
      return;
    }
    generationInFlight.current = true;
    setGeneratingGapId(gap.id);
    setGenerationStartedAt(Date.now());
    setGenerationResults((prev) => {
      const next = { ...prev };
      delete next[gap.id];
      return next;
    });
    api
      .generateOpportunity(gap.id, generationCutoff || undefined, project?.id)
      .then((report) => {
        setGenerationResults((prev) => ({ ...prev, [gap.id]: { phase: 'done', report } }));
        loadOpportunities();
      })
      .catch((err: Error) =>
        setGenerationResults((prev) => ({
          ...prev,
          [gap.id]: { phase: 'error', message: err.message || 'Opportunity generation failed.' },
        })),
      )
      .finally(() => {
        generationInFlight.current = false;
        setGeneratingGapId(null);
        setGenerationStartedAt(null);
      });
  };

  if (loading) {
    return <LoadingState message="Loading research gaps..." submessage="Querying stored gaps and opportunities" minHeight="400px" />;
  }

  if (error) {
    return (
      <div className="u-stack-lg">
        <ErrorBanner title="Could Not Load Research Gaps" message={error} onRetry={() => loadAll(true)} />
      </div>
    );
  }

  const discovering = discoveryStartedAt !== null;
  const selectedResult = selectedGap ? generationResults[selectedGap.id] : undefined;
  const selectedExisting = selectedGap ? opportunitiesByGap.get(selectedGap.id) ?? [] : [];
  const gapCutoff =
    selectedGap?.metadata && typeof selectedGap.metadata.temporal_cutoff === 'string'
      ? selectedGap.metadata.temporal_cutoff
      : null;

  return (
    <div className="view-container animate-fade-in">
      <div className="view-header">
        <div>
          <div className="u-row">
            <h1 className="view-title">Research Gaps</h1>
            <span className="app-badge badge-neutral">{gaps.length} Total</span>
          </div>
          <p className="view-subtitle">
            Gaps are discovered by clustering extracted research statements and validated by the LLM. Select a
            VALIDATED gap to generate a research opportunity from it.
          </p>
        </div>
        <button
          type="button"
          onClick={() => loadAll(true)}
          className="btn btn-secondary"
          disabled={discovering || generatingGapId !== null}
          title="Reload gaps and opportunities"
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Discovery control */}
      <div className="card">
        <div className="card-header">
          <div className="u-row">
            <Radar size={16} className="tone-cyan" />
            <h3 className="card-title">Discover gaps from the corpus</h3>
          </div>
        </div>
        <div className="card-body u-stack">
          <div className="u-row u-wrap u-between">
            <p className="u-text-sm u-text-secondary discovery-copy">
              Runs semantic gap discovery over this project's research statements only (its papers must have
              statements extracted first). New gaps are added to this project; existing gaps are kept.
            </p>
            <div className="u-row u-wrap">
              <label className="form-field form-field-inline">
                <span className="form-label">Statement cutoff (optional)</span>
                <input
                  type="date"
                  value={discoveryCutoff}
                  onChange={(e) => setDiscoveryCutoff(e.target.value)}
                  className="form-input"
                  disabled={discovering}
                />
              </label>
              <button type="button" className="btn btn-primary" onClick={runDiscovery} disabled={discovering}>
                {discovering ? <Loader2 size={14} className="spin" /> : <Radar size={14} />}
                <span>{discovering ? 'Discovering…' : 'Run gap discovery'}</span>
              </button>
            </div>
          </div>
          {discovering && (
            <div className="inline-pending" role="status">
              <Loader2 size={16} className="spin" />
              <span>
                Gap discovery is running on the server ({formatElapsed(discoveryElapsed)} elapsed). The
                request is synchronous, so results appear only when it finishes.
              </span>
            </div>
          )}
          {discoveryState?.phase === 'error' && (
            <div className="action-result is-error" role="alert">
              <p className="action-result-title">Gap discovery failed</p>
              <p className="u-text-sm">{discoveryState.message}</p>
            </div>
          )}
          {discoveryState?.phase === 'done' && (
            <div className={`action-result ${discoveryState.created.length > 0 ? 'is-success' : 'is-warning'}`} role="status">
              <p className="action-result-title">
                Discovery finished: {discoveryState.created.length} gap{discoveryState.created.length === 1 ? '' : 's'} returned
                {' '}({discoveryState.created.filter((g) => g.status === 'VALIDATED').length} validated).
              </p>
              {discoveryState.created.length === 0 && (
                <p className="u-text-sm">
                  No gaps were produced. Gap discovery needs enough related statements across papers; add and analyze
                  more papers, then try again.
                </p>
              )}
            </div>
          )}
        </div>
      </div>

      {gaps.length === 0 ? (
        <EmptyState
          title="No research gaps yet"
          description="Add papers from Search Papers, extract their research statements, then run gap discovery above."
          icon={<Sparkles size={24} />}
          action={
            <button type="button" className="btn btn-secondary" onClick={() => onNavigate('#/search')}>
              Go to Search Papers
            </button>
          }
        />
      ) : (
        <div className="corpus-layout">
          {/* Gap list */}
          <div className="corpus-papers-pane">
            <div className="card">
              <div className="card-header">
                <div className="u-row">
                  <Sparkles size={16} className="tone-amber" />
                  <h3 className="card-title">Gaps</h3>
                </div>
                <span className="card-header-meta">{filteredGaps.length} / {gaps.length}</span>
              </div>
              <div className="pane-toolbar">
                {GAP_FILTERS.map((f) => (
                  <button
                    type="button"
                    key={f}
                    onClick={() => setFilter(f)}
                    className={`filter-pill-sm ${filter === f ? 'active' : ''}`}
                    aria-pressed={filter === f}
                  >
                    {gapFilterLabel(f)} ({counts[f]})
                  </button>
                ))}
              </div>
              <div className="corpus-paper-list">
                {filteredGaps.length === 0 && <p className="empty-note card-body">No gaps have this status.</p>}
                {filteredGaps.map((gap) => {
                  const isSelected = gap.id === selectedGapId;
                  const generated = opportunitiesByGap.get(gap.id)?.length ?? 0;
                  return (
                    <button
                      type="button"
                      key={gap.id}
                      onClick={() => setSelectedGapId(gap.id)}
                      className={`corpus-paper-item ${isSelected ? 'selected' : ''}`}
                      aria-current={isSelected ? 'true' : undefined}
                    >
                      <span className="u-row u-between corpus-paper-item-head">
                        <Badge label={String(gap.status)} variant={getGapStatusBadgeVariant(gap.status)} />
                        {generatingGapId === gap.id && (
                          <span className="u-row u-row-tight u-text-xs tone-indigo">
                            <Loader2 size={12} className="spin" /> Generating
                          </span>
                        )}
                      </span>
                      <span className="corpus-paper-item-title u-block clamp-3">{gap.gap_statement}</span>
                      <span className="corpus-paper-item-meta">
                        <span>{gap.evidence_count} evidence</span>
                        <span>·</span>
                        <span>{gap.supporting_paper_ids.length} papers</span>
                        {generated > 0 && (
                          <>
                            <span>·</span>
                            <span>{generated} opportunit{generated === 1 ? 'y' : 'ies'}</span>
                          </>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Selected gap detail + generation */}
          <div className="corpus-detail-pane">
            {!selectedGap ? (
              <EmptyState title="Select a gap" description="Choose a gap from the list to see its evidence and generate an opportunity." />
            ) : (
              <div className="u-stack-lg">
                <div className="card">
                  <div className="card-header">
                    <div className="u-row u-wrap">
                      <Badge label={`Gap: ${selectedGap.status}`} variant={getGapStatusBadgeVariant(selectedGap.status)} />
                      <Badge label="Model-Validated" variant="model-generated" />
                    </div>
                    <span className="u-mono u-text-xs u-text-tertiary">Gap ID: {selectedGap.id}</span>
                  </div>
                  <div className="card-body u-stack">
                    <h2 className="paper-detail-title">{selectedGap.gap_statement}</h2>
                    <div>
                      <span className="label-caps">Research problem</span>
                      <p className="prose-body"><RecordedValue value={selectedGap.research_problem} /></p>
                    </div>
                    <div>
                      <span className="label-caps">Validation reasoning</span>
                      <p className="prose-small"><RecordedValue value={selectedGap.validation_reasoning} /></p>
                      <p className="u-text-xs u-text-tertiary">
                        Model-reported confidence: <RecordedValue value={displayPercent(selectedGap.validation_confidence)} />
                      </p>
                    </div>
                    {(selectedGap.limitations.length > 0 || selectedGap.future_work.length > 0) && (
                      <div className="grid-2-cols">
                        <div>
                          <span className="label-caps">Recurring limitations</span>
                          {selectedGap.limitations.length === 0 ? (
                            <p className="empty-note">None recorded.</p>
                          ) : (
                            <ul className="plain-list">
                              {selectedGap.limitations.map((l, i) => <li key={i}>{l}</li>)}
                            </ul>
                          )}
                        </div>
                        <div>
                          <span className="label-caps">Future work</span>
                          {selectedGap.future_work.length === 0 ? (
                            <p className="empty-note">None recorded.</p>
                          ) : (
                            <ul className="plain-list">
                              {selectedGap.future_work.map((l, i) => <li key={i}>{l}</li>)}
                            </ul>
                          )}
                        </div>
                      </div>
                    )}
                    <div className="section-divider u-stack">
                      <span className="label-caps">
                        Supporting evidence ({selectedGap.supporting_statement_ids.length} statements from{' '}
                        {selectedGap.supporting_paper_ids.length} papers)
                      </span>
                      {selectedGap.supporting_paper_ids.length > 0 && (
                        <ul className="plain-list">
                          {selectedGap.supporting_paper_ids.map((pid) => (
                            <li key={pid}>
                              <a className="btn-link btn-link-inline" href={`#/papers/${pid}`}>
                                {paperTitles.get(pid) ?? `Paper ${pid.slice(0, 8)}…`}
                              </a>
                            </li>
                          ))}
                        </ul>
                      )}
                      {selectedGap.supporting_statement_ids.length > 0 ? (
                        <div className="u-row u-wrap">
                          {selectedGap.supporting_statement_ids.map((sid, i) => (
                            <button
                              type="button"
                              key={sid}
                              className="chip-button"
                              onClick={() => onInspectEvidence(sid)}
                              title="Inspect statement and source passage"
                            >
                              <Bookmark size={12} />
                              <span>Statement {i + 1}</span>
                            </button>
                          ))}
                        </div>
                      ) : (
                        <p className="empty-note">No supporting statements are linked to this gap.</p>
                      )}
                    </div>
                  </div>
                </div>

                {/* Generation */}
                <div className="card">
                  <div className="card-header">
                    <div className="u-row">
                      <Wand2 size={16} className="tone-indigo" />
                      <h3 className="card-title">Generate research opportunity</h3>
                    </div>
                  </div>
                  <div className="card-body u-stack">
                    {selectedGap.status !== 'VALIDATED' ? (
                      <div className="inline-note">
                        <Info size={16} />
                        <span>
                          Only VALIDATED gaps can be used for generation. This gap is {selectedGap.status}.
                        </span>
                      </div>
                    ) : (
                      <>
                        <p className="u-text-sm u-text-secondary">
                          Runs the generation pipeline synchronously: drafts a proposal, retrieves prior art from the
                          corpus, validates it and refines it up to 2 times. Accepted proposals (VALIDATED or REFINED)
                          are tracked in the opportunity lifecycle.
                        </p>
                        <div className="u-row u-wrap">
                          <label className="form-field form-field-inline">
                            <span className="form-label">Prior-art cutoff (optional)</span>
                            <input
                              type="date"
                              value={generationCutoff}
                              onChange={(e) => setGenerationCutoff(e.target.value)}
                              className="form-input"
                              disabled={generatingGapId !== null}
                            />
                          </label>
                          {!generationCutoff && gapCutoff && (
                            <span className="u-text-xs u-text-tertiary">Defaults to this gap's cutoff ({gapCutoff}).</span>
                          )}
                        </div>
                        {selectedExisting.length > 0 && (
                          <div className="inline-note">
                            <Info size={16} />
                            <span>
                              {selectedExisting.length} opportunit{selectedExisting.length === 1 ? 'y was' : 'ies were'} already
                              generated from this gap. Generating again creates another record.
                            </span>
                          </div>
                        )}
                        <div>
                          <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => generate(selectedGap)}
                            disabled={generatingGapId !== null}
                          >
                            {generatingGapId === selectedGap.id ? <Loader2 size={14} className="spin" /> : <Wand2 size={14} />}
                            <span>
                              {generatingGapId === selectedGap.id
                                ? 'Generating…'
                                : generatingGapId !== null
                                  ? 'Another generation is running'
                                  : 'Generate opportunity'}
                            </span>
                          </button>
                        </div>
                      </>
                    )}

                    {generatingGapId === selectedGap.id && (
                      <div className="inline-pending" role="status">
                        <Loader2 size={16} className="spin" />
                        <span>
                          Generation is running on the server ({formatElapsed(generationElapsed)} elapsed). This
                          usually takes several minutes with a local model; keep this tab open.
                        </span>
                      </div>
                    )}

                    {selectedResult?.phase === 'error' && (
                      <div className="action-result is-error" role="alert">
                        <p className="action-result-title">Generation failed — nothing was confirmed as saved</p>
                        <p className="u-text-sm">{selectedResult.message}</p>
                      </div>
                    )}
                    {selectedResult?.phase === 'done' && (
                      <GenerationResult
                        report={selectedResult.report}
                        onOpen={(id) => onNavigate(`#/opportunities/${id}`)}
                      />
                    )}

                    {opportunitiesError && (
                      <p className="u-text-xs tone-rose">Existing opportunities could not be loaded: {opportunitiesError}</p>
                    )}
                    {selectedExisting.length > 0 && (
                      <div className="section-divider u-stack">
                        <span className="label-caps">Opportunities generated from this gap</span>
                        <ul className="existing-opps-list">
                          {selectedExisting.map((o) => (
                            <li key={o.id}>
                              <button
                                type="button"
                                className="existing-opp-item"
                                onClick={() => onNavigate(`#/opportunities/${o.id}`)}
                              >
                                <span className="u-row u-wrap">
                                  <Badge label={String(o.status)} variant={getGenerationStatusBadgeVariant(o.status)} />
                                  <span className="u-text-xs u-text-tertiary">
                                    {o.created_at ? new Date(o.created_at).toLocaleString() : 'Date not recorded'}
                                  </span>
                                </span>
                                <span className="u-text-sm clamp-3">{o.proposed_research_direction}</span>
                                <ArrowRight size={14} className="existing-opp-arrow" />
                              </button>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
