import React, { useCallback, useEffect, useState, useMemo } from 'react';
import {
  Search,
  ArrowRight,
  Sparkles,
  Layers,
  Clock,
  RefreshCw
} from 'lucide-react';
import { LIFECYCLE_STATUSES, type LifecycleStatus, type OpportunitySummary } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { EmptyState } from '../common/EmptyState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getLifecycleBadgeVariant } from '../common/badgeVariants';
import { NotRecorded } from '../common/NotRecorded';

type StatusFilter = 'ALL' | LifecycleStatus | 'NOT_TRACKED';

const STATUS_FILTERS: readonly StatusFilter[] = ['ALL', ...LIFECYCLE_STATUSES, 'NOT_TRACKED'];

function statusFilterLabel(filter: StatusFilter): string {
  if (filter === 'ALL') return 'All Opportunities';
  if (filter === 'NOT_TRACKED') return 'Not tracked';
  return filter;
}

interface OpportunitiesExplorerViewProps {
  onNavigate: (route: string) => void;
}

export const OpportunitiesExplorerView: React.FC<OpportunitiesExplorerViewProps> = ({ onNavigate }) => {
  const [opportunities, setOpportunities] = useState<OpportunitySummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('ALL');

  const fetchOpportunities = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getOpportunities()
      .then((data) => setOpportunities(data))
      .catch((err: Error) => setError(err.message || 'Failed to load research opportunities'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchOpportunities();
  }, [fetchOpportunities]);

  const filtered = useMemo(() => {
    return opportunities.filter((opp) => {
      // Status filter
      if (statusFilter === 'NOT_TRACKED') {
        if (opp.lifecycle_status) return false;
      } else if (statusFilter !== 'ALL') {
        if (opp.lifecycle_status !== statusFilter) return false;
      }
      // Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const dir = (opp.proposed_research_direction || '').toLowerCase();
        const gap = (opp.identified_gap || '').toLowerCase();
        const prob = (opp.research_problem || '').toLowerCase();
        return dir.includes(q) || gap.includes(q) || prob.includes(q);
      }
      return true;
    });
  }, [opportunities, statusFilter, searchQuery]);

  if (loading) {
    return <LoadingState message="Loading research opportunities..." minHeight="400px" />;
  }

  if (error) {
    return (
      <div className="u-stack-lg">
        <ErrorBanner
          title="Could Not Load Opportunities"
          message={error}
          onRetry={fetchOpportunities}
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
            <h1 className="view-title">Research Opportunities Explorer</h1>
            <span className="app-badge badge-neutral">{opportunities.length} Total</span>
          </div>
          <p className="view-subtitle">
            Model-generated research proposals derived from research gap records, with temporal evidence from later-published papers where recorded.
          </p>
        </div>
        <button
          onClick={fetchOpportunities}
          className="btn btn-secondary"
          title="Refresh opportunities list"
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Filter Toolbar */}
      <div className="toolbar-container">
        {/* Search */}
        <div className="search-input-wrapper">
          <Search className="search-icon" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search proposals, gap statements, or problems..."
            className="search-input"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="search-clear-btn">
              Clear
            </button>
          )}
        </div>

        {/* Lifecycle Status Filter Tabs */}
        <div className="filter-pill-group">
          {STATUS_FILTERS.map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className={`filter-pill ${statusFilter === st ? 'active' : ''}`}
              aria-pressed={statusFilter === st}
            >
              {statusFilterLabel(st)}
            </button>
          ))}
        </div>
      </div>

      {/* Opportunities List */}
      {filtered.length === 0 ? (
        <EmptyState
          title="No Matching Opportunities"
          description={searchQuery ? `No opportunity proposals matched "${searchQuery}".` : 'No opportunities match the selected lifecycle filter.'}
          action={
            <button onClick={() => { setSearchQuery(''); setStatusFilter('ALL'); }} className="btn btn-secondary">
              Reset Filters
            </button>
          }
        />
      ) : (
        <div className="opportunities-grid">
          {filtered.map((opp) => {
            const lifecycleStatus = opp.lifecycle_status ?? null;
            return (
              <div key={opp.id} className="opportunity-card">
                {/* Card Top Row: Badges & ID */}
                <div className="opportunity-card-top">
                  <div className="u-row u-wrap">
                    <Badge
                      label={`Lifecycle: ${lifecycleStatus ?? 'Not tracked'}`}
                      variant={getLifecycleBadgeVariant(lifecycleStatus)}
                    />
                    <Badge
                      label={`Pipeline: ${opp.status || 'Not recorded'}`}
                      variant="neutral"
                    />
                    <Badge
                      label="Model-Generated Proposal"
                      variant="model-generated"
                    />
                  </div>
                  <span className="u-mono u-text-xs u-text-tertiary">
                    ID: {opp.id.slice(0, 8)}...
                  </span>
                </div>

                {/* Proposed Direction Title */}
                <h3 className="opportunity-direction-title">
                  {opp.proposed_research_direction}
                </h3>

                {/* Associated Research Gap */}
                <div className="opportunity-gap-snippet">
                  <div className="u-row u-row-tight excerpt-header tone-amber">
                    <Sparkles size={14} />
                    <span>Identified Research Gap</span>
                  </div>
                  {opp.identified_gap || opp.research_problem ? (
                    <p className="prose-small u-italic clamp-3">
                      "{opp.identified_gap || opp.research_problem}"
                    </p>
                  ) : (
                    <p className="prose-small"><NotRecorded /></p>
                  )}
                </div>

                {/* Card Footer: Metadata & Inspect Button */}
                <div className="opportunity-card-footer">
                  <div className="opportunity-meta-stats">
                    <span className="meta-stat-pill" title="Temporal evidence events linked to this proposal">
                      <Clock size={14} className="tone-indigo" />
                      <span>{opp.evidence_count} Temporal Evidence Records</span>
                    </span>
                    <span className="meta-stat-pill" title="Supporting statements linked to this proposal">
                      <Layers size={14} />
                      <span>{opp.supporting_statement_count} Supporting Statements</span>
                    </span>
                  </div>

                  <button
                    onClick={() => onNavigate(`#/opportunities/${opp.id}`)}
                    className="btn btn-primary btn-sm"
                  >
                    <span>View Detail & Evolution</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
