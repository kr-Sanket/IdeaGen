import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Cpu,
  Calendar,
  Layers,
  Filter,
  AlertTriangle,
  RefreshCw,
  FileText,
  Play,
} from 'lucide-react';
import type { ConfigurationFingerprint, RunMetadata, ResearchProject, ResearchRun, WorkspaceStats } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getScreeningDecisionBadgeVariant } from '../common/badgeVariants';
import { NotRecorded } from '../common/NotRecorded';
import { displayValue } from '../../format';
import { openRunProgress, registerMonitoredRun, useRunMonitor } from '../../workflow/runMonitor';

interface RunHistoryViewProps {
  project: ResearchProject;
  onNavigate: (route: string) => void;
}

const GENERIC_CAVEAT =
  'Results are model-generated and have not been validated by human experts; see run counts below.';

export const RunHistoryView: React.FC<RunHistoryViewProps> = ({ project, onNavigate }) => {
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [projectStats, setProjectStats] = useState<WorkspaceStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [projectRuns, setProjectRuns] = useState<ResearchRun[]>([]);
  const { monitoredRun } = useRunMonitor();

  const fetchProjectRuns = useCallback(() => {
    if (!project?.id) return;
    api.getProjectRuns(project.id)
      .then((runs) => setProjectRuns(runs))
      .catch((err) => console.warn('Failed to fetch project runs:', err));
  }, [project?.id]);

  const fetchMetadata = useCallback(() => {
    setLoading(true);
    setError(null);
    // Metric cards are project-scoped; the experiment fingerprint below is workspace-wide.
    api.getWorkspaceStats(project?.id)
      .then(setProjectStats)
      .catch(() => setProjectStats(null));
    api.getRunMetadata()
      .then((data) => setMetadata(data))
      .catch((err: Error) => setError(err.message || 'Failed to fetch experiment run metadata.'))
      .finally(() => setLoading(false));
  }, [project?.id]);

  useEffect(() => {
    fetchMetadata();
    fetchProjectRuns();
  }, [fetchMetadata, fetchProjectRuns]);

  const rankedPapers = useMemo(() => {
    const ranked = metadata?.relevance_screening?.ranked_papers;
    if (!Array.isArray(ranked)) return [];
    return [...ranked].sort((a, b) => a.rank - b.rank);
  }, [metadata]);

  if (loading) {
    return <LoadingState message="Loading experiment run history & configuration..." minHeight="400px" />;
  }

  if (error || !metadata) {
    return (
      <div className="u-stack-lg">
        <ErrorBanner
          title="Could Not Load Run History"
          message={error || 'No metadata returned from experiment database.'}
          onRetry={fetchMetadata}
        />
      </div>
    );
  }

  const fp: ConfigurationFingerprint | null = metadata.run_fingerprint;
  const screening = metadata.relevance_screening;
  const counts = metadata.counts;

  return (
    <div className="view-container animate-fade-in">
      {/* View Header */}
      <div className="view-header">
        <div>
          <div className="u-row" style={{ gap: '0.625rem', marginBottom: '0.25rem' }}>
            <span className="badge badge-primary">Run History</span>
            <span className="badge badge-default">Project: {project.name}</span>
          </div>
          <h1 className="view-title">Experiment Run History &amp; Diagnostics</h1>
          <p className="view-subtitle">
            Auditable configuration fingerprint, temporal evaluation cutoffs, relevance screening decisions, and model parameters.
          </p>
        </div>
        <button
          onClick={fetchMetadata}
          className="btn btn-secondary"
          title="Refresh run history"
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Limitations Callout */}
      <div className="callout callout-warning">
        <div className="callout-icon-col">
          <AlertTriangle size={20} />
        </div>
        <div className="callout-content">
          <h4 className="callout-title">Experimental Scope &amp; Methodological Limitations</h4>
          {metadata.metric_interpretation_note && (
            <p className="callout-text">{metadata.metric_interpretation_note}</p>
          )}
          <p className={metadata.metric_interpretation_note ? 'callout-subtext' : 'callout-text'}>
            {GENERIC_CAVEAT}
          </p>
        </div>
      </div>

      {/* Persistent Research Runs Table */}
      <div className="card u-stack-md">
        <div className="card-header-clean u-row u-between">
          <div className="u-row" style={{ gap: '0.5rem', alignItems: 'center' }}>
            <Play size={16} className="text-primary" />
            <h3 className="card-title">Project Research Runs ({projectRuns.length})</h3>
          </div>
          <span className="badge badge-default" style={{ fontSize: '0.6875rem' }}>
            Auditable Execution History
          </span>
        </div>

        {projectRuns.length === 0 ? (
          <p className="text-tertiary" style={{ fontSize: '0.8125rem', padding: '0.5rem 0' }}>
            No research runs executed for this project yet. Go to Search &amp; Ingest to select papers and launch a run.
          </p>
        ) : (
          <div className="card-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Status</th>
                  <th>Stages</th>
                  <th>Papers</th>
                  <th>Started At</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {projectRuns.map((r) => {
                  const stagesList = r.stages && r.stages.length > 0 ? r.stages : (r.stage_progress?.stages || []);
                  const completedStages = stagesList.filter((s: any) => s.status === 'COMPLETED').length;
                  const isActiveMonitored = monitoredRun?.runId === r.id;
                  return (
                    <tr key={r.id} className={isActiveMonitored ? 'is-active-run-row' : undefined}>
                      <td className="font-mono text-primary font-semibold">
                        <div className="u-row" style={{ gap: '0.375rem', alignItems: 'center' }}>
                          <span>{r.id}</span>
                          {isActiveMonitored && (
                            <span className="badge badge-primary" style={{ fontSize: '0.625rem', padding: '0.125rem 0.375rem' }}>
                              Active
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            r.status === 'COMPLETED'
                              ? 'badge-success'
                              : r.status === 'RUNNING' || r.status === 'PENDING'
                                ? 'badge-primary'
                                : r.status === 'FAILED' || r.status === 'INTERRUPTED'
                                  ? 'badge-danger'
                                  : r.status === 'AWAITING_REVIEW' || r.status === 'PARTIAL_SUCCESS'
                                    ? 'badge-warning'
                                    : 'badge-default'
                          }`}
                          style={{ fontSize: '0.6875rem' }}
                        >
                          {r.status.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="text-secondary" style={{ fontSize: '0.8125rem' }}>
                        {completedStages} / {stagesList.length} completed
                      </td>
                      <td className="text-secondary" style={{ fontSize: '0.8125rem' }}>
                        {r.papers.length} paper(s)
                      </td>
                      <td className="font-mono text-tertiary" style={{ fontSize: '0.75rem' }}>
                        {r.started_at ? new Date(r.started_at).toLocaleString() : '—'}
                      </td>
                      <td>
                        <button
                          type="button"
                          className="btn btn-secondary btn-xs"
                          onClick={() => {
                            if (r.status === 'RUNNING' || r.status === 'PENDING') {
                              registerMonitoredRun(r.id, project.id, project.name);
                            }
                            openRunProgress(r.id, project.id);
                          }}
                        >
                          Inspect Run &amp; Logs
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Primary Metrics Grid */}
      <div className="stat-cards-grid">
        <div className="stat-card u-clickable" onClick={() => onNavigate(`#/projects/${project.id}/papers`)} role="button" tabIndex={0}>
          <div className="stat-card-header">
            <span className="stat-card-label">Corpus Papers</span>
            <FileText className="stat-card-icon stat-icon-primary" size={18} />
          </div>
          <div className="stat-card-value">{projectStats ? projectStats.total_papers : '—'}</div>
          <div className="stat-card-sub">
            {projectStats ? `${projectStats.analyzed_papers} with extracted statements · this project` : 'Not loaded'}
          </div>
        </div>

        <div className="stat-card u-clickable" onClick={() => onNavigate(`#/projects/${project.id}/papers?tab=corpus`)} role="button" tabIndex={0}>
          <div className="stat-card-header">
            <span className="stat-card-label">Extracted Statements</span>
            <Layers className="stat-card-icon stat-icon-cyan" size={18} />
          </div>
          <div className="stat-card-value">{projectStats ? projectStats.total_statements : '—'}</div>
          <div className="stat-card-sub">From this project's papers</div>
        </div>

        <div className="stat-card u-clickable" onClick={() => onNavigate(`#/projects/${project.id}/gaps`)} role="button" tabIndex={0}>
          <div className="stat-card-header">
            <span className="stat-card-label">Discovered Gaps</span>
            <Activity className="stat-card-icon stat-icon-amber" size={18} />
          </div>
          <div className="stat-card-value">{projectStats ? projectStats.total_gaps : '—'}</div>
          <div className="stat-card-sub">
            {projectStats ? `${projectStats.validated_gaps} validated · ${projectStats.candidate_gaps} candidate` : 'Not loaded'}
          </div>
        </div>

        <div className="stat-card u-clickable" onClick={() => onNavigate(`#/projects/${project.id}/opportunities`)} role="button" tabIndex={0}>
          <div className="stat-card-header">
            <span className="stat-card-label">Opportunities</span>
            <Activity className="stat-card-icon stat-icon-emerald" size={18} />
          </div>
          <div className="stat-card-value">{projectStats ? projectStats.total_opportunities : '—'}</div>
          <div className="stat-card-sub">
            {projectStats ? `${projectStats.validated_opportunities} validated or refined` : 'Not loaded'}
          </div>
        </div>
      </div>

      <div className="section-header-row">
        <div>
          <h2 className="section-title">Workspace-Wide Experiment Configuration</h2>
          <p className="section-subtitle">
            Recorded by the controlled experiment runner and shared by all projects; not specific to {project.name}.
          </p>
        </div>
      </div>

      {/* Two Column Layout for Configuration & Execution Timeline */}
      <div className="grid-2col">
        {/* Run Configuration Fingerprint Card */}
        <div className="card u-stack-md">
          <div className="card-header-clean">
            <h3 className="card-title">
              <Cpu size={16} className="text-primary" style={{ display: 'inline', marginRight: '0.5rem' }} />
              Configuration Fingerprint
            </h3>
            <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
              Immutable Run Config
            </span>
          </div>

          <div className="u-stack-sm">
            <div className="detail-row">
              <span className="detail-label">LLM Provider</span>
              <span className="detail-value">{displayValue(fp?.llm_provider)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">LLM Model</span>
              <span className="detail-value font-mono text-primary font-semibold">{displayValue(fp?.llm_model)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Embedding Provider</span>
              <span className="detail-value">{displayValue(fp?.embedding_provider)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Embedding Model</span>
              <span className="detail-value font-mono">{displayValue(fp?.embedding_model)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Embedding Dimension</span>
              <span className="detail-value font-mono">{displayValue(fp?.embedding_dimension)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Max Refinement Iterations</span>
              <span className="detail-value font-mono">{displayValue(fp?.max_refinement_iterations)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Passage Selection</span>
              <span className="detail-value">
                {fp?.passage_selection ? (
                  <span>
                    max {fp.passage_selection.max_passages} passages, max {fp.passage_selection.max_total_words} words
                  </span>
                ) : (
                  <NotRecorded label="Default passage windowing" />
                )}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Gap Similarity Threshold</span>
              <span className="detail-value font-mono">
                {displayValue(fp?.gap_discovery?.similarity_threshold)}
              </span>
            </div>
          </div>
        </div>

        {/* Temporal Cutoffs & Execution Rules */}
        <div className="card u-stack-md">
          <div className="card-header-clean">
            <h3 className="card-title">
              <Calendar size={16} className="text-cyan" style={{ display: 'inline', marginRight: '0.5rem' }} />
              Temporal Evaluation Cutoffs
            </h3>
            <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
              Stage 5 Evaluation
            </span>
          </div>

          <div className="u-stack-sm">
            <div className="detail-row">
              <span className="detail-label">Baseline Cutoff Date</span>
              <span className="detail-value font-mono text-emerald font-semibold">
                {displayValue(metadata.baseline_cutoff)}
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Stage 5 Temporal Evaluation</span>
              <span className="detail-value">
                <span className={`badge ${metadata.stage_5_enabled ? 'badge-success' : 'badge-default'}`}>
                  {metadata.stage_5_enabled ? 'Enabled' : 'Disabled'}
                </span>
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Evaluation Windows</span>
              <span className="detail-value">
                {Array.isArray(metadata.evaluation_cutoffs) && metadata.evaluation_cutoffs.length > 0 ? (
                  <div className="u-row" style={{ gap: '0.25rem', flexWrap: 'wrap' }}>
                    {metadata.evaluation_cutoffs.map((c, i) => (
                      <span key={i} className="badge badge-info" style={{ fontSize: '0.6875rem' }}>
                        {c}
                      </span>
                    ))}
                  </div>
                ) : (
                  <NotRecorded label="No staged evaluation cutoffs recorded" />
                )}
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Database Connection</span>
              <span className="detail-value font-mono text-secondary">
                {displayValue(metadata.database_status)}
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Evolution Evidence Records</span>
              <span className="detail-value font-semibold">
                {counts?.evidence_records ?? 0} persisted
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Lifecycle History Records</span>
              <span className="detail-value font-semibold">
                {counts?.history_records ?? 0} persisted
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Relevance Screening Table if configured */}
      {screening && (
        <div className="card u-stack-md">
          <div className="card-header-clean">
            <h3 className="card-title">
              <Filter size={16} className="text-amber" style={{ display: 'inline', marginRight: '0.5rem' }} />
              Relevance Screening Decisions ({rankedPapers.length} papers)
            </h3>
            <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
              Threshold: {screening.threshold}
            </span>
          </div>

          <div className="card-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Paper Title</th>
                  <th>Date</th>
                  <th>Similarity</th>
                  <th>Decision</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {rankedPapers.map((paper) => (
                  <tr key={paper.paper_id}>
                    <td className="font-mono text-tertiary">#{paper.rank}</td>
                    <td className="font-semibold text-primary">{paper.title || paper.filename}</td>
                    <td className="font-mono text-tertiary" style={{ fontSize: '0.8125rem' }}>
                      {paper.publication_date || '—'}
                    </td>
                    <td className="font-mono">{paper.score.toFixed(3)}</td>
                    <td>
                      <Badge
                        label={paper.decision}
                        variant={getScreeningDecisionBadgeVariant(paper.decision)}
                      />
                    </td>
                    <td className="text-secondary" style={{ fontSize: '0.75rem' }}>{paper.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
