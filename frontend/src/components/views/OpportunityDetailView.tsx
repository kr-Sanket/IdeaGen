import React, { useEffect, useState } from 'react';
import {
  ArrowLeft,
  Sparkles,
  Clock,
  Layers,
  FileText,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  ShieldCheck,
  Calendar,
  RefreshCw,
  ChevronRight,
  Bookmark,
  Info
} from 'lucide-react';
import type {
  OpportunityReport,
  OpportunityEvidenceItem,
  OpportunityHistoryItem
} from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getGenerationStatusBadgeVariant, getLifecycleBadgeVariant, getRelationshipBadgeVariant } from '../common/badgeVariants';
import { NotRecorded, RecordedValue } from '../common/NotRecorded';
import { displayPercent } from '../../format';
import { EvolvePanel } from '../workflow/EvolvePanel';

interface OpportunityDetailViewProps {
  opportunityId: string;
  /** Active project; the backend returns 404 for an opportunity outside it. */
  projectId?: string;
  onNavigate: (route: string) => void;
  onInspectEvidence: (statementId: string) => void;
}

export const OpportunityDetailView: React.FC<OpportunityDetailViewProps> = ({
  opportunityId,
  projectId,
  onNavigate,
  onInspectEvidence,
}) => {
  const [report, setReport] = useState<OpportunityReport | null>(null);
  const [evidenceList, setEvidenceList] = useState<OpportunityEvidenceItem[]>([]);
  const [historyList, setHistoryList] = useState<OpportunityHistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  // Full load shows the loading state; a silent refresh (after an action) keeps the page mounted.
  const fetchData = React.useCallback((silent: boolean = false) => {
    if (silent) {
      setRefreshing(true);
      setRefreshError(null);
    } else {
      setLoading(true);
      setError(null);
    }
    Promise.all([
      api.getOpportunityReport(opportunityId, projectId),
      api.getOpportunityEvidence(opportunityId, projectId),
      api.getOpportunityHistory(opportunityId, projectId),
    ])
      .then(([rep, evs, hists]) => {
        setReport(rep);
        // Sort evidence by publication date / cutoff ascending
        const sortedEvs = [...evs].sort((a, b) => {
          const dateA = a.paper_publication_date || a.evaluation_cutoff || '';
          const dateB = b.paper_publication_date || b.evaluation_cutoff || '';
          return dateA.localeCompare(dateB);
        });
        setEvidenceList(sortedEvs);
        setHistoryList(hists);
      })
      .catch(async (err: Error) => {
        const message = err.message || 'Failed to fetch opportunity report and evolution history.';
        if (silent) {
          setRefreshError(message);
          return;
        }
        // The report endpoint only accepts generation record ids. If this id is a lifecycle
        // opportunity id (e.g. from an evolution outcome), resolve it via the list endpoint.
        try {
          const list = await api.getOpportunities();
          const match = list.find((o) => o.lifecycle_opportunity_id === opportunityId);
          if (match) {
            onNavigate(`#/opportunities/${match.id}`);
            return;
          }
        } catch {
          // fall through to the original error
        }
        setError(message);
      })
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  }, [opportunityId, projectId, onNavigate]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (loading) {
    return <LoadingState message="Loading opportunity proposal and temporal evolution history..." minHeight="450px" />;
  }

  if (error || !report) {
    return (
      <div className="u-stack-lg">
        <button onClick={() => onNavigate('#/opportunities')} className="btn btn-secondary btn-sm">
          <ArrowLeft size={14} />
          <span>Back to Opportunities</span>
        </button>
        <ErrorBanner
          title="Could Not Load Opportunity Detail"
          message={error || 'Opportunity report was not found.'}
          onRetry={() => fetchData()}
        />
      </div>
    );
  }


  const currentLifecycleStatus = report.lifecycle_status ?? null;
  const linkedPaperIds = new Set(evidenceList.map((ev) => ev.paper_id));
  const supportingPapers = report.supporting_papers ?? [];
  const allSupportingPapersBaseline =
    supportingPapers.length > 0 && supportingPapers.every((p) => p.temporal_role === 'BASELINE');

  return (
    <div className="view-container animate-fade-in">
      {/* Top Breadcrumb & Navigation */}
      <div className="breadcrumb-nav">
        <button
          onClick={() => onNavigate('#/opportunities')}
          className="breadcrumb-back-btn"
        >
          <ArrowLeft size={14} />
          <span>Opportunities Explorer</span>
        </button>
        <ChevronRight size={14} className="tone-faint" />
        <span className="u-mono u-text-sm u-text-secondary">
          ID: {opportunityId.slice(0, 8)}...
        </span>
      </div>

      {/* Header Card */}
      <div className="opportunity-detail-header-card">
        <div className="u-row u-wrap u-between detail-header-top">
          <div className="u-row u-wrap">
            <Badge
              label={`Lifecycle Status: ${currentLifecycleStatus ?? 'Not tracked'}`}
              variant={getLifecycleBadgeVariant(currentLifecycleStatus)}
            />
            <Badge
              label={`Generation Status: ${report.status || 'Not recorded'}`}
              variant={getGenerationStatusBadgeVariant(report.status)}
            />
            <Badge
              label="Model-Generated Proposal"
              variant="model-generated"
            />
          </div>
          <button
            onClick={() => fetchData(true)}
            className="btn btn-secondary btn-sm"
            title="Refresh details"
            disabled={refreshing}
          >
            <RefreshCw size={14} className={refreshing ? 'spin' : undefined} />
            <span>{refreshing ? 'Refreshing…' : 'Refresh'}</span>
          </button>
        </div>

        <h1 className="detail-title">
          {report.proposed_research_direction}
        </h1>

        <div className="u-row u-wrap u-row-loose detail-id-row">
          <span>Generation record ID: {report.opportunity_id}</span>
          <span>
            Gap ID:{' '}
            <a className="btn-link btn-link-inline" href={`#/gaps/${report.gap_id}`}>{report.gap_id}</a>
          </span>
          <span>Lifecycle ID: {report.lifecycle_opportunity_id ?? 'None (not tracked)'}</span>
          <span>Evidence records: {evidenceList.length}</span>
          <span>Transitions: {historyList.length}</span>
          <span>Generation Model: <RecordedValue value={report.generation_model} /></span>
        </div>
      </div>

      {/* Main 2-Column Responsive Layout */}
      <div className="layout-grid-detail">
        {/* Left Column: Research Proposal & Baseline Evidence */}
        <div className="u-stack-lg">
          {/* Associated Research Gap */}
          <div className="card">
            <div className="card-header">
              <div className="u-row">
                <Sparkles size={16} className="tone-amber" />
                <h3 className="card-title">Associated Literature Gap</h3>
              </div>
              {allSupportingPapersBaseline && (
                <Badge
                  label="Pre-Cutoff Baseline"
                  variant="temporal-baseline"
                  title="All supporting papers have temporal_role BASELINE"
                />
              )}
            </div>
            <div className="card-body u-stack">
              <div>
                <span className="label-caps">Research Problem</span>
                <p className="prose-body">
                  <RecordedValue value={report.research_problem} />
                </p>
              </div>
              <div>
                <span className="label-caps">Identified Gap</span>
                {report.identified_gap ? (
                  <p className="prose-quote">"{report.identified_gap}"</p>
                ) : (
                  <p className="prose-body"><NotRecorded /></p>
                )}
              </div>
              {report.why_existing_work_is_insufficient && (
                <div>
                  <span className="label-caps">Why Existing Baseline Work is Insufficient</span>
                  <p className="prose-small">
                    {report.why_existing_work_is_insufficient}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Proposal Architecture & Rationale */}
          <div className="card">
            <div className="card-header">
              <div className="u-row">
                <FileText size={16} className="tone-indigo" />
                <h3 className="card-title">Technical Proposal Architecture</h3>
              </div>
              <Badge label="Model-Generated" variant="model-generated" />
            </div>
            <div className="card-body u-stack-md">
              <div>
                <h4 className="label-caps">Reasoning & Hypothesized Mechanism</h4>
                <p className="prose-body"><RecordedValue value={report.reasoning} /></p>
              </div>

              <div>
                <h4 className="label-caps">Proposed Technical Approach</h4>
                <p className="prose-body"><RecordedValue value={report.technical_approach} /></p>
              </div>

              <div>
                <h4 className="label-caps">Expected Research Contribution</h4>
                <p className="prose-body"><RecordedValue value={report.expected_contribution} /></p>
              </div>

              {report.differentiation && (
                <div>
                  <h4 className="label-caps">Differentiation from Prior Art</h4>
                  <p className="prose-small">
                    {report.differentiation}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Feasibility & Novelty Heuristic Signals */}
          <div className="card">
            <div className="card-header">
              <div className="u-row">
                <ShieldCheck size={16} className="tone-emerald" />
                <h3 className="card-title">Feasibility & Novelty Signals</h3>
              </div>
              <Badge label="Automated Heuristic" variant="model-generated" />
            </div>
            <div className="card-body u-stack-md">
              <div className="grid-2-cols">
                <div className="sub-stat-box">
                  <span className="sub-stat-label">Feasibility Score</span>
                  <span className="sub-stat-val u-mono tone-emerald">
                    <RecordedValue value={displayPercent(report.feasibility?.feasibility_score)} />
                  </span>
                  <span className="sub-stat-note">
                    Complexity: <RecordedValue value={report.feasibility?.implementation_complexity} />
                  </span>
                </div>
                <div className="sub-stat-box">
                  <span className="sub-stat-label">Model Confidence</span>
                  <span className="sub-stat-val u-mono tone-indigo">
                    <RecordedValue value={displayPercent(report.confidence)} />
                  </span>
                  <span className="sub-stat-note">
                    Model-reported confidence (not externally validated)
                  </span>
                </div>
              </div>

              <div className="info-table">
                <div className="info-row">
                  <span className="info-key">Required Data</span>
                  <span className="info-value"><RecordedValue value={report.feasibility?.required_data} /></span>
                </div>
                <div className="info-row">
                  <span className="info-key">Required Compute</span>
                  <span className="info-value"><RecordedValue value={report.feasibility?.required_compute} /></span>
                </div>
                <div className="info-row">
                  <span className="info-key">Experimental Setup</span>
                  <span className="info-value"><RecordedValue value={report.feasibility?.experimental_setup} /></span>
                </div>
              </div>
            </div>
          </div>

          {/* Supporting Baseline Evidence Statements */}
          <div className="card">
            <div className="card-header">
              <div className="u-row">
                <Layers size={16} className="tone-indigo" />
                <h3 className="card-title">Supporting Statements</h3>
              </div>
              <span className="card-header-meta">
                {report.supporting_evidence.length} Statements
              </span>
            </div>
            <div className="card-body">
              {report.supporting_evidence.length === 0 ? (
                <p className="empty-note">No direct supporting statement details recorded.</p>
              ) : (
                <div className="u-stack">
                  {report.supporting_evidence.map((ev, idx) => (
                    <div key={`${ev.statement_id}-${idx}`} className="supporting-statement-card">
                      <div className="u-row u-between supporting-statement-head">
                        <span className="u-text-sm u-text-strong">
                          <RecordedValue value={ev.paper_title} missingLabel="Paper title not recorded" />
                        </span>
                        <button
                          onClick={() => onInspectEvidence(ev.statement_id)}
                          className="btn-link btn-link-inline"
                          title="Inspect linked source passage"
                        >
                          <Bookmark size={12} className="tone-indigo" />
                          <span>Inspect Source</span>
                        </button>
                      </div>
                      <p className="supporting-statement-text">
                        "{ev.statement_text}"
                      </p>
                      <div className="u-row u-text-xs u-text-secondary u-mono">
                        <span className="app-badge badge-neutral">{ev.statement_type}</span>
                        {ev.section && <span>Section: {ev.section}</span>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Evolution control & Temporal Evolution Timeline */}
        <div className="u-stack-lg">
          {refreshError && (
            <ErrorBanner
              title="Could Not Refresh Evidence"
              message={refreshError}
              onRetry={() => fetchData(true)}
            />
          )}
          {report.lifecycle_opportunity_id ? (
            <EvolvePanel
              projectId={projectId}
              lifecycleId={report.lifecycle_opportunity_id}
              opportunityCreatedAt={report.created_at}
              linkedPaperIds={linkedPaperIds}
              onEvolved={() => fetchData(true)}
            />
          ) : (
            <div className="inline-note">
              <Info size={16} />
              <span>
                This proposal has generation status {report.status || 'not recorded'} and is not tracked in the
                opportunity lifecycle (only VALIDATED and REFINED proposals are), so it cannot be evolved.
              </span>
            </div>
          )}
          <div className="card">
            <div className="card-header">
              <div className="u-row">
                <Clock size={16} className="tone-indigo" />
                <h3 className="card-title">Temporal Evolution Evidence Timeline</h3>
              </div>
              <span className="card-header-meta">
                {evidenceList.length} Evidence Records
              </span>
            </div>
            <div className="card-body">
              <p className="timeline-intro">
                Evidence records from later literature, each with a model-assigned relationship
                (SUPPORTS, ADDRESSES, CONTRADICTS, EXTENDS, UNRELATED). Lifecycle transitions are
                applied by deterministic code and shown where recorded.
              </p>

              {evidenceList.length === 0 ? (
                <div className="empty-timeline-box">
                  <Info size={20} />
                  <p>No temporal evolution evidence was recorded for this opportunity.</p>
                </div>
              ) : (
                <div className="evolution-timeline">
                  {evidenceList.map((ev, idx) => {
                    // Check if a state transition occurred linked to this evidence
                    const transition = historyList.find((h) => h.evidence_id === ev.id);

                    return (
                      <div key={ev.id || idx} className="evolution-timeline-node">
                        <div className="timeline-node-marker">
                          <span className="timeline-bullet" />
                          {idx < evidenceList.length - 1 && <span className="timeline-connector-line" />}
                        </div>

                        <div className="timeline-card">
                          {/* Event Meta Header */}
                          <div className="u-row u-wrap u-between timeline-card-head">
                            <div className="u-row">
                              <Badge
                                label={`Classification: ${ev.relationship || 'Not recorded'}`}
                                variant={getRelationshipBadgeVariant(ev.relationship)}
                              />
                              <Badge label="Model-Classified" variant="model-generated" />
                            </div>
                            <span className="u-row u-row-tight u-text-sm u-mono tone-cyan">
                              <Calendar size={14} />
                              <span>
                                <RecordedValue
                                  value={ev.paper_publication_date || ev.evaluation_cutoff}
                                  missingLabel="Date not recorded"
                                />
                              </span>
                            </span>
                          </div>

                          {/* Paper Title & Batch Window */}
                          <h4 className="timeline-paper-title">
                            {ev.paper_title || `Paper ${ev.paper_id.slice(0, 8)}... (title not recorded)`}
                          </h4>
                          <div className="u-row u-wrap timeline-chip-row">
                            {ev.batch_id && (
                              <span className="chip-mono">Batch: {ev.batch_id}</span>
                            )}
                            {ev.evaluation_cutoff && (
                              <span className="chip-mono">Cutoff Window: {ev.evaluation_cutoff}</span>
                            )}
                          </div>

                          {/* Source Excerpt with server-side verification result */}
                          {ev.supporting_text ? (
                            <EvidenceExcerpt text={ev.supporting_text} verified={ev.verified} />
                          ) : (
                            <p className="empty-note">
                              No supporting passage excerpt was captured for this evidence record.
                            </p>
                          )}

                          {/* Lifecycle State Transition */}
                          <div className="timeline-transition-footer">
                            {transition ? (
                              <div className="u-row u-wrap u-text-sm">
                                <span className="u-text-secondary">Recorded Transition:</span>
                                <Badge
                                  label={transition.previous_status}
                                  variant={getLifecycleBadgeVariant(transition.previous_status)}
                                />
                                <span className="u-text-tertiary">→</span>
                                <Badge
                                  label={transition.new_status}
                                  variant={getLifecycleBadgeVariant(transition.new_status)}
                                />
                                {transition.created_at && (
                                  <span className="u-text-xs u-text-tertiary u-mono">
                                    ({new Date(transition.created_at).toLocaleDateString()})
                                  </span>
                                )}
                              </div>
                            ) : (
                              <div className="u-text-xs u-text-secondary">
                                <span className="u-text-strong">No lifecycle transition recorded</span>{' '}
                                for this evidence record. Classifier explanations are not stored for
                                non-transitioning evaluations.
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

interface EvidenceExcerptProps {
  text: string;
  verified: boolean | null | undefined;
}

/**
 * Renders an evidence excerpt labelled strictly by the backend `verified` flag
 * (computed by is_evidence_grounded_for_paper against stored source passages).
 */
const EvidenceExcerpt: React.FC<EvidenceExcerptProps> = ({ text, verified }) => {
  let boxClass = 'timeline-excerpt-box is-unknown';
  let headerClass = 'u-row u-row-tight excerpt-header u-text-tertiary';
  let icon = <HelpCircle size={14} />;
  let label = 'Verification not available';

  if (verified === true) {
    boxClass = 'timeline-excerpt-box';
    headerClass = 'u-row u-row-tight excerpt-header tone-emerald';
    icon = <CheckCircle2 size={14} />;
    label = 'Excerpt verified against stored source text';
  } else if (verified === false) {
    boxClass = 'timeline-excerpt-box is-unverified';
    headerClass = 'u-row u-row-tight excerpt-header tone-rose';
    icon = <AlertTriangle size={14} />;
    label = 'Excerpt could not be matched to stored source text';
  }

  return (
    <div className={boxClass}>
      <div className={headerClass}>
        {icon}
        <span>{label}</span>
      </div>
      <p className="excerpt-text">"{text}"</p>
    </div>
  );
};
