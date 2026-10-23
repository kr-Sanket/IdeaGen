import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GitBranch, Loader2, Search, RefreshCw } from 'lucide-react';
import type { OpportunityEvolveResponse, PaperItem } from '../../types';
import { api } from '../../api/client';
import { Badge } from '../common/Badge';
import { getLifecycleBadgeVariant, getRelationshipBadgeVariant } from '../common/badgeVariants';
import { formatElapsed, useElapsedSeconds } from '../../hooks/useElapsedSeconds';

const MAX_PAPERS_PER_REQUEST = 100;

interface EvolvePanelProps {
  /** Lifecycle Opportunity id (not the generation record id). */
  lifecycleId: string;
  /** ISO timestamp of the opportunity's creation; papers published earlier are skipped by the server. */
  opportunityCreatedAt: string | null;
  /** Papers already linked as evidence to this lifecycle opportunity. */
  linkedPaperIds: ReadonlySet<string>;
  /** Called after the backend confirms an evolve request, so the caller can reload evidence/history. */
  onEvolved: () => void;
  /** Active project: candidate papers come from its corpus and the server enforces the scope. */
  projectId?: string;
}

type EvolveState =
  | { phase: 'done'; result: OpportunityEvolveResponse }
  | { phase: 'error'; message: string };

export const EvolvePanel: React.FC<EvolvePanelProps> = ({
  lifecycleId,
  opportunityCreatedAt,
  linkedPaperIds,
  onEvolved,
  projectId,
}) => {
  const [papers, setPapers] = useState<PaperItem[]>([]);
  const [papersLoading, setPapersLoading] = useState<boolean>(true);
  const [papersError, setPapersError] = useState<string | null>(null);
  const [filterText, setFilterText] = useState<string>('');
  const [onlyDateEligible, setOnlyDateEligible] = useState<boolean>(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [cutoffDate, setCutoffDate] = useState<string>('');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [state, setState] = useState<EvolveState | null>(null);
  const inFlight = useRef<boolean>(false);
  const elapsed = useElapsedSeconds(startedAt);

  const createdDate = opportunityCreatedAt ? opportunityCreatedAt.slice(0, 10) : null;

  const loadPapers = useCallback(() => {
    setPapersLoading(true);
    setPapersError(null);
    api
      .getPapers(projectId)
      .then(setPapers)
      .catch((err: Error) => setPapersError(err.message || 'Could not load corpus papers.'))
      .finally(() => setPapersLoading(false));
  }, [projectId]);

  useEffect(() => {
    loadPapers();
  }, [loadPapers]);

  const paperTitles = useMemo(() => new Map(papers.map((p) => [p.id, p.title || p.source_id])), [papers]);

  /** Mirrors the server's skip rules so the user can see which papers will be evaluated. */
  const skipReason = useCallback(
    (p: PaperItem): string | null => {
      if (linkedPaperIds.has(p.id)) return 'Already linked as evidence';
      if (createdDate) {
        if (!p.publication_date) return 'Undated';
        if (p.publication_date < createdDate) return `Published before ${createdDate}`;
      }
      if (cutoffDate && p.publication_date && p.publication_date > cutoffDate) return `Published after cutoff`;
      return null;
    },
    [linkedPaperIds, createdDate, cutoffDate],
  );

  const visiblePapers = useMemo(() => {
    const q = filterText.trim().toLowerCase();
    return papers.filter((p) => {
      if (onlyDateEligible && skipReason(p)) return false;
      if (!q) return true;
      return (p.title || '').toLowerCase().includes(q) || p.source_id.toLowerCase().includes(q);
    });
  }, [papers, filterText, onlyDateEligible, skipReason]);

  const eligibleCount = useMemo(() => papers.filter((p) => !skipReason(p)).length, [papers, skipReason]);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else if (next.size < MAX_PAPERS_PER_REQUEST) next.add(id);
      return next;
    });
  };

  const submit = () => {
    if (inFlight.current || selected.size === 0) return;
    const ids = Array.from(selected);
    inFlight.current = true;
    setState(null);
    setStartedAt(Date.now());
    api
      .evolveOpportunity(lifecycleId, { paper_ids: ids, ...(cutoffDate ? { cutoff_date: cutoffDate } : {}) }, projectId)
      .then((result) => {
        setState({ phase: 'done', result });
        setSelected(new Set());
        onEvolved();
      })
      .catch((err: Error) => setState({ phase: 'error', message: err.message || 'Evolution request failed.' }))
      .finally(() => {
        inFlight.current = false;
        setStartedAt(null);
      });
  };

  const running = startedAt !== null;

  return (
    <div className="card">
      <div className="card-header">
        <div className="u-row">
          <GitBranch size={16} className="tone-purple" />
          <h3 className="card-title">Evolve with corpus papers</h3>
        </div>
        <button type="button" className="btn btn-secondary btn-sm" onClick={loadPapers} disabled={papersLoading || running}>
          <RefreshCw size={12} />
          <span>Reload papers</span>
        </button>
      </div>
      <div className="card-body u-stack">
        <p className="u-text-sm u-text-secondary">
          Each selected paper is classified against this opportunity by the LLM (SUPPORTS, ADDRESSES, CONTRADICTS,
          EXTENDS or UNRELATED). Grounded results are stored as evidence and the lifecycle transition is applied by code.
          The server skips papers that are already linked, undated, or published before the opportunity was created
          {createdDate ? ` (${createdDate})` : ''}.
        </p>

        <div className="u-row u-wrap">
          <div className="search-input-wrapper-sm">
            <Search className="search-icon-sm" />
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter papers by title or source ID"
              className="search-input-sm"
              aria-label="Filter corpus papers"
            />
          </div>
          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={onlyDateEligible}
              onChange={(e) => setOnlyDateEligible(e.target.checked)}
            />
            <span>Only date-eligible, unlinked papers ({eligibleCount})</span>
          </label>
          <label className="form-field form-field-inline">
            <span className="form-label">Cutoff (optional)</span>
            <input
              type="date"
              value={cutoffDate}
              onChange={(e) => setCutoffDate(e.target.value)}
              className="form-input"
              disabled={running}
            />
          </label>
        </div>

        {papersError && <p className="u-text-sm tone-rose">{papersError}</p>}
        {papersLoading ? (
          <div className="inline-pending"><Loader2 size={14} className="spin" /> <span>Loading corpus papers…</span></div>
        ) : visiblePapers.length === 0 ? (
          <p className="empty-note">
            {onlyDateEligible
              ? `No corpus papers are eligible: evolution needs papers published on or after ${createdDate ?? 'the opportunity creation date'} that are not already linked. Add newer papers from Search Papers, or untick the filter to see all papers.`
              : 'No corpus papers match this filter.'}
          </p>
        ) : (
          <ul className="paper-pick-list" aria-label="Corpus papers">
            {visiblePapers.map((p) => {
              const reason = skipReason(p);
              const checked = selected.has(p.id);
              const linked = linkedPaperIds.has(p.id);
              return (
                <li key={p.id}>
                  <label className={`paper-pick-item ${checked ? 'is-checked' : ''}`}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={running || linked || (!checked && selected.size >= MAX_PAPERS_PER_REQUEST)}
                      onChange={() => toggle(p.id)}
                    />
                    <span className="paper-pick-body">
                      <span className="u-text-sm u-text-strong">{p.title || p.source_id}</span>
                      <span className="u-row u-wrap u-text-xs u-text-tertiary">
                        <span>{p.publication_date ?? 'No date'}</span>
                        <span>· {p.passage_count} passages</span>
                        {reason && <span className="tone-amber">· {reason}{linked ? '' : ' — server will skip'}</span>}
                      </span>
                    </span>
                  </label>
                </li>
              );
            })}
          </ul>
        )}

        <div className="u-row u-wrap u-between">
          <span className="u-text-xs u-text-tertiary">
            {selected.size} selected (max {MAX_PAPERS_PER_REQUEST})
          </span>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={running || selected.size === 0}>
            {running ? <Loader2 size={14} className="spin" /> : <GitBranch size={14} />}
            <span>{running ? 'Evaluating…' : selected.size > 0 ? `Evaluate ${selected.size} paper${selected.size === 1 ? '' : 's'}` : 'Select papers to evaluate'}</span>
          </button>
        </div>

        {running && (
          <div className="inline-pending" role="status">
            <Loader2 size={16} className="spin" />
            <span>
              Classification is running on the server ({formatElapsed(elapsed)} elapsed); one LLM call per eligible
              paper. Results appear when the request completes.
            </span>
          </div>
        )}

        {state?.phase === 'error' && (
          <div className="action-result is-error" role="alert">
            <p className="action-result-title">Evolution request failed — no change confirmed</p>
            <p className="u-text-sm">{state.message}</p>
          </div>
        )}

        {state?.phase === 'done' && (
          <EvolveResultSummary result={state.result} paperTitles={paperTitles} />
        )}
      </div>
    </div>
  );
};

interface EvolveResultSummaryProps {
  result: OpportunityEvolveResponse;
  paperTitles: Map<string, string>;
}

const EvolveResultSummary: React.FC<EvolveResultSummaryProps> = ({ result, paperTitles }) => {
  const changed = result.initial_status !== result.final_status;
  const tone = result.failed_count > 0 ? 'is-warning' : result.processed_count > 0 ? 'is-success' : 'is-warning';
  const title = (id: string) => paperTitles.get(id) ?? `Paper ${id.slice(0, 8)}…`;

  return (
    <div className={`action-result ${tone}`} role="status">
      <div className="u-row u-wrap">
        <span className="u-text-sm u-text-strong">Lifecycle:</span>
        <Badge label={result.initial_status} variant={getLifecycleBadgeVariant(result.initial_status)} />
        <span>→</span>
        <Badge label={result.final_status} variant={getLifecycleBadgeVariant(result.final_status)} />
        <span className="u-text-xs u-text-secondary">{changed ? 'status changed' : 'status unchanged'}</span>
      </div>
      <div className="count-row">
        <span>Processed: {result.processed_count}</span>
        <span>Skipped: {result.skipped_count}</span>
        <span>Failed: {result.failed_count}</span>
      </div>
      {result.processed_count === 0 && result.failed_count === 0 && (
        <p className="u-text-sm">
          No paper was evaluated. Selected papers were skipped by the server (already linked, undated, outside the date
          window, or rejected during evidence grounding).
        </p>
      )}
      {result.events.length > 0 && (
        <ul className="evolution-change-list">
          {result.events.map((ev) => (
            <li key={`${ev.paper_id}-${ev.evidence_id ?? ''}`} className="u-row u-wrap u-text-xs">
              <span className="u-text-strong">{title(ev.paper_id)}</span>
              <Badge label={ev.relationship} variant={getRelationshipBadgeVariant(ev.relationship)} />
              <Badge label={ev.previous_status} variant={getLifecycleBadgeVariant(ev.previous_status)} />
              <span>→</span>
              <Badge label={ev.new_status} variant={getLifecycleBadgeVariant(ev.new_status)} />
              {!ev.state_changed && <span className="u-text-tertiary">(no transition)</span>}
            </li>
          ))}
        </ul>
      )}
      {result.errors.length > 0 && (
        <ul className="evolution-change-list">
          {result.errors.map((er) => (
            <li key={er.paper_id} className="u-text-xs tone-rose">
              {title(er.paper_id)}: {er.reason}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
