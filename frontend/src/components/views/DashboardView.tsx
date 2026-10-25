import React, { useCallback, useEffect, useState, useMemo } from 'react';
import {
  FolderPlus,
  Compass,
  BookOpen,
  Lightbulb,
  GitBranch,
  ArrowRight,
  Sparkles,
  Layers,
  Clock,
  CheckCircle2,
  Play,
  Trash2,
} from 'lucide-react';
import type {
  ResearchProject,
  WorkspaceStats,
  OpportunitySummary,
  PaperItem,
  ProjectRecord,
  ResearchRun,
  ProjectDeleteResponse,
} from '../../types';
import { api } from '../../api/client';
import { deleteProject as defaultDeleteProject } from '../../projects/projectStore';
import { LoadingState } from '../common/LoadingState';
import { ErrorBanner } from '../common/ErrorBanner';
import { CreateProjectModal } from './CreateProjectModal';
import { NewRunModal } from './NewRunModal';
import { DeleteProjectModal } from './DeleteProjectModal';

interface DashboardViewProps {
  projects: ResearchProject[];
  activeProjectId: string;
  onSelectProject: (projectId: string) => void;
  onCreateProject: (params: { name: string; topic: string; description?: string; temporalCutoff?: string }) => Promise<ResearchProject>;
  onDeleteProject?: (projectId: string) => Promise<ProjectDeleteResponse>;
  onNavigate: (route: string) => void;
}

interface RecentActivityEntry {
  id: string;
  timestamp: string;
  type: 'EVOLUTION' | 'INGESTION' | 'OPPORTUNITY';
  title: string;
  detail: string;
  badge: string;
  badgeVariant: 'success' | 'info' | 'warning' | 'default';
  route: string;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  projects,
  activeProjectId,
  onSelectProject,
  onCreateProject,
  onDeleteProject,
  onNavigate,
}) => {
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [opportunities, setOpportunities] = useState<OpportunitySummary[]>([]);
  const [recentPapers, setRecentPapers] = useState<PaperItem[]>([]);
  const [records, setRecords] = useState<Record<string, ProjectRecord>>({});
  const [latestRuns, setLatestRuns] = useState<Record<string, ResearchRun | null>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [isCreateModalOpen, setIsCreateModalOpen] = useState<boolean>(false);
  const [isRunModalOpen, setIsRunModalOpen] = useState<boolean>(false);
  const [runModalProject, setRunModalProject] = useState<ResearchProject | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState<boolean>(false);
  const [projectToDelete, setProjectToDelete] = useState<ResearchProject | null>(null);
  const [successNotification, setSuccessNotification] = useState<string | null>(null);

  const projectIdsKey = projects.map((p) => p.id).join('|');

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const ids = projectIdsKey ? projectIdsKey.split('|') : [];
    // Workspace totals are global; the activity feed belongs to the active project; each project
    // card uses its own server-side counts and latest run.
    const [statsRes, oppsRes, papersRes, recordsRes, ...runsRes] = await Promise.allSettled([
      api.getWorkspaceStats(),
      api.getOpportunities(activeProjectId),
      api.getPapers(activeProjectId),
      api.listProjects(),
      ...ids.map((id) => api.getProjectRuns(id)),
    ]);
    const failures: string[] = [];
    const reason = (r: PromiseRejectedResult) => (r.reason instanceof Error ? r.reason.message : String(r.reason));

    if (statsRes.status === 'fulfilled') setStats(statsRes.value);
    else { setStats(null); failures.push(reason(statsRes)); }
    setOpportunities(oppsRes.status === 'fulfilled' ? oppsRes.value : []);
    setRecentPapers(papersRes.status === 'fulfilled' ? papersRes.value : []);
    if (recordsRes.status === 'fulfilled') {
      setRecords(Object.fromEntries(recordsRes.value.map((r) => [r.id, r])));
    } else {
      setRecords({});
      failures.push(reason(recordsRes));
    }
    const runs: Record<string, ResearchRun | null> = {};
    ids.forEach((id, i) => {
      const r = runsRes[i];
      // A project unknown to the server (404) simply has no runs yet.
      runs[id] = r && r.status === 'fulfilled' ? (r.value[0] ?? null) : null;
    });
    setLatestRuns(runs);
    if (failures.length) setError(failures[0]);
    setLoading(false);
  }, [activeProjectId, projectIdsKey]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  // Compute honest recent research activity from persisted records
  const recentActivity: RecentActivityEntry[] = useMemo(() => {
    const list: RecentActivityEntry[] = [];

    // Opportunity evolution & creation activity
    opportunities.forEach((opp) => {
      if (!opp.created_at) return;
      if (opp.lifecycle_status === 'PARTIALLY_ADDRESSED' || opp.lifecycle_status === 'STRENGTHENED') {
        list.push({
          id: `act_evolve_${opp.id}`,
          timestamp: opp.created_at,
          type: 'EVOLUTION',
          title: `Opportunity Lifecycle Status: ${opp.lifecycle_status}`,
          detail: `Direction: "${opp.proposed_research_direction?.slice(0, 95)}..." with ${opp.evidence_count} evidence records.`,
          badge: opp.lifecycle_status,
          badgeVariant: 'success',
          route: `#/projects/${activeProjectId}/opportunities/${opp.id}`,
        });
      } else {
        list.push({
          id: `act_opp_${opp.id}`,
          timestamp: opp.created_at,
          type: 'OPPORTUNITY',
          title: `Research Opportunity Formulated`,
          detail: `Confidence: ${Math.round(opp.confidence * 100)}% · Derived from Gap ${opp.gap_id?.slice(0, 8)}...`,
          badge: opp.status,
          badgeVariant: 'info',
          route: `#/projects/${activeProjectId}/opportunities/${opp.id}`,
        });
      }
    });

    // Papers added to the corpus (the API returns them newest first by ingestion time)
    const sortedPapers = recentPapers.filter((p) => p.created_at).slice(0, 4);
    sortedPapers.forEach((paper) => {
      list.push({
        id: `act_paper_${paper.id}`,
        timestamp: paper.created_at as string,
        type: 'INGESTION',
        title: `Paper Ingested into Corpus`,
        detail: `"${paper.title}" (${paper.publication_date || 'Date unrecorded'})`,
        badge: paper.is_baseline ? 'BASELINE' : 'LATER_EVAL',
        badgeVariant: paper.is_baseline ? 'default' : 'warning',
        route: `#/projects/${activeProjectId}/papers?tab=corpus`,
      });
    });

    return list.sort((a, b) => b.timestamp.localeCompare(a.timestamp)).slice(0, 6);
  }, [opportunities, recentPapers, activeProjectId]);

  const handleOpenProject = (projectId: string) => {
    onSelectProject(projectId);
    onNavigate(`#/projects/${projectId}/overview`);
  };

  const handleOpenRunModal = (project: ResearchProject) => {
    setRunModalProject(project);
    setIsRunModalOpen(true);
  };

  if (loading) {
    return <LoadingState message="Loading research workspace dashboard..." minHeight="400px" />;
  }

  return (
    <div className="view-container animate-fade-in">
      {/* Workspace Dashboard Header */}
      <div className="view-header">
        <div>
          <div className="u-row" style={{ gap: '0.625rem', marginBottom: '0.25rem' }}>
            <span className="badge badge-primary">Personal Workspace</span>
            <span className="badge badge-default">Private Research</span>
          </div>
          <h1 className="view-title">Research Projects Dashboard</h1>
          <p className="view-subtitle">
            Manage your empirical research projects, formulate evidenced hypotheses, and track longitudinal opportunity evolution.
          </p>
        </div>
        <div className="u-row" style={{ gap: '0.75rem' }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setIsCreateModalOpen(true)}
            id="btn-create-project"
          >
            <FolderPlus size={16} />
            <span>New Project</span>
          </button>
        </div>
      </div>

      {successNotification && (
        <div
          className="callout callout-info animate-fade-in"
          style={{
            marginBottom: '1rem',
            background: 'var(--emerald-light, rgba(16, 185, 129, 0.08))',
            borderColor: 'var(--emerald-border, rgba(16, 185, 129, 0.25))',
          }}
          role="status"
        >
          <div className="callout-icon-col" style={{ color: 'var(--emerald-accent, #059669)' }}>
            <CheckCircle2 size={16} />
          </div>
          <div className="callout-content" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-primary)' }}>{successNotification}</span>
            <button
              type="button"
              className="btn-icon"
              style={{ padding: '0.25rem', border: 'none', background: 'transparent', cursor: 'pointer' }}
              onClick={() => setSuccessNotification(null)}
              aria-label="Dismiss notification"
            >
              <span style={{ fontSize: '1.2rem', lineHeight: 1, color: 'var(--text-secondary)' }}>&times;</span>
            </button>
          </div>
        </div>
      )}

      {error && (
        <ErrorBanner
          title="Could Not Refresh Workspace Stats"
          message={error}
          onRetry={loadData}
        />
      )}

      {/* Global Workspace Summary Cards */}
      <div className="stat-cards-grid">
        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Research Projects</span>
            <span className="stat-card-icon stat-icon-cyan">
              <Layers size={18} />
            </span>
          </div>
          <div className="stat-card-value">{projects.length}</div>
          <div className="stat-card-sub">
            {projects.filter((p) => p.isDefaultBenchmark).length} active benchmark · {projects.filter((p) => !p.isDefaultBenchmark).length} user-configured
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Corpus Literature</span>
            <span className="stat-card-icon stat-icon-primary">
              <BookOpen size={18} />
            </span>
          </div>
          <div className="stat-card-value">{stats ? stats.total_papers : '—'}</div>
          <div className="stat-card-sub">
            {stats ? `${stats.total_statements} statements extracted · all projects` : 'Not loaded'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Discovered Gaps</span>
            <span className="stat-card-icon stat-icon-amber">
              <Lightbulb size={18} />
            </span>
          </div>
          <div className="stat-card-value">{stats ? stats.total_gaps : '—'}</div>
          <div className="stat-card-sub">
            {stats ? `${stats.validated_gaps} validated · ${stats.candidate_gaps} candidate · all projects` : 'Not loaded'}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-card-label">Generated Opportunities</span>
            <span className="stat-card-icon stat-icon-emerald">
              <Compass size={18} />
            </span>
          </div>
          <div className="stat-card-value">{stats ? stats.total_opportunities : '—'}</div>
          <div className="stat-card-sub">
            {stats ? `${stats.validated_opportunities} validated or refined · all projects` : 'Not loaded'}
          </div>
        </div>
      </div>

      {/* Project Cards Section */}
      <div className="u-stack-md">
        <div className="section-header-row">
          <div>
            <h2 className="section-title">Your Research Projects</h2>
            <p className="section-subtitle">
              Projects maintain independent research contexts, literature collections, and opportunity lifecycle pipelines.
            </p>
          </div>
          <span className="text-tertiary" style={{ fontSize: '0.8125rem' }}>
            Showing {projects.length} project{projects.length !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="project-cards-grid">
          {projects.map((proj) => {
            const isBenchmark = proj.isDefaultBenchmark;
            const isActive = proj.id === activeProjectId;

            // Project-scoped counts from the backend; '—' when the server has not reported them.
            const record = records[proj.id];
            const count = (v: number | undefined) => (record ? String(v ?? 0) : '—');
            const latestRun = latestRuns[proj.id] ?? null;
            const papers = record?.paper_count ?? 0;
            const gaps = record?.gap_count ?? 0;

            let attentionState: { statusText: string; badgeVariant: 'success' | 'warning' | 'danger' | 'default' | 'info'; note: string };
            if (!record) {
              attentionState = { statusText: 'Not synced', badgeVariant: 'warning', note: 'The backend has not reported this project yet.' };
            } else if (!latestRun) {
              attentionState = { statusText: 'No runs yet', badgeVariant: 'default', note: `${papers} corpus paper(s) · ${record.selection_count} queued for a run` };
            } else {
              const variant = latestRun.status === 'AWAITING_REVIEW' || latestRun.status === 'COMPLETED'
                ? 'success'
                : latestRun.status === 'FAILED' || latestRun.status === 'INTERRUPTED'
                  ? 'danger'
                  : latestRun.status === 'PARTIAL_SUCCESS' ? 'warning' : 'info';
              attentionState = {
                statusText: `Last run: ${latestRun.status.replace(/_/g, ' ')}`,
                badgeVariant: variant,
                note: latestRun.error_message || `${record.run_count} run(s) · ${latestRun.summary?.failed_papers ?? 0} failed paper(s) in the latest run`,
              };
            }

            let clearNextAction: { desc: string };
            if (papers === 0) {
              clearNextAction = { desc: 'Search OpenAlex and add papers to build this project\'s corpus.' };
            } else if (latestRun && ['FAILED', 'INTERRUPTED', 'PARTIAL_SUCCESS'].includes(latestRun.status)) {
              clearNextAction = { desc: 'Open Run History to inspect the failure and retry the run.' };
            } else if (gaps > 0) {
              clearNextAction = { desc: 'Review discovered gaps and their evidence; generate an opportunity from a validated gap.' };
            } else {
              clearNextAction = { desc: 'Start a research run to extract statements and discover gaps from the corpus.' };
            }

            return (
              <div
                key={proj.id}
                className={`project-card ${isActive ? 'is-active-project' : ''}`}
                onClick={() => onSelectProject(proj.id)}
                role="button"
                tabIndex={0}
                aria-pressed={isActive}
                aria-label={`Select project ${proj.name}`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSelectProject(proj.id);
                  }
                }}
              >
                {/* Card Header */}
                <div className="project-card-header">
                  <div>
                    <div className="u-row" style={{ gap: '0.5rem', marginBottom: '0.375rem' }}>
                      {isBenchmark && (
                        <span className="badge badge-primary" style={{ fontSize: '0.625rem' }}>
                          Primary Benchmark
                        </span>
                      )}
                      {isActive && (
                        <span className="badge badge-success" style={{ fontSize: '0.625rem' }}>
                          Active Project
                        </span>
                      )}
                      <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
                        Cutoff: {proj.temporalCutoff || 'None'}
                      </span>
                    </div>
                    <h3 className="project-card-title">{proj.name}</h3>
                  </div>
                </div>

                {/* Topic / Description */}
                <p className="project-card-topic">
                  <strong>Topic:</strong> {proj.topic}
                </p>

                {proj.description && (
                  <p className="project-card-desc">{proj.description}</p>
                )}

                {/* Real Metrics Strip */}
                <div className="project-card-metrics">
                  <div className="project-metric">
                    <span className="project-metric-value">{count(record?.paper_count)}</span>
                    <span className="project-metric-label">Papers</span>
                  </div>
                  <div className="project-metric">
                    <span className="project-metric-value">{count(record?.gap_count)}</span>
                    <span className="project-metric-label">Gaps</span>
                  </div>
                  <div className="project-metric">
                    <span className="project-metric-value">{count(record?.opportunity_count)}</span>
                    <span className="project-metric-label">Opportunities</span>
                  </div>
                  <div className="project-metric">
                    <span className="project-metric-value">{count(record?.run_count)}</span>
                    <span className="project-metric-label">Runs</span>
                  </div>
                </div>

                {/* Runs Needing Attention */}
                <div className="project-attention-box">
                  <div className="u-row" style={{ justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                    <span className="attention-label">
                      <Clock size={13} style={{ marginRight: '0.25rem' }} />
                      Run Status &amp; Attention:
                    </span>
                    <span className={`badge badge-${attentionState.badgeVariant}`} style={{ fontSize: '0.625rem' }}>
                      {attentionState.statusText}
                    </span>
                  </div>
                  <p className="attention-note">{attentionState.note}</p>
                </div>

                {/* Clear Next Action Box */}
                <div className="project-next-action-box">
                  <div className="next-action-header">
                    <Sparkles size={14} className="text-primary" />
                    <span className="next-action-title">Recommended Next Action:</span>
                  </div>
                  <p className="next-action-desc">{clearNextAction.desc}</p>
                </div>

                {/* Card Action Buttons */}
                <div className="project-card-actions">
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleOpenProject(proj.id);
                    }}
                    style={{ flex: 1 }}
                  >
                    <span>Open Project</span>
                    <ArrowRight size={14} />
                  </button>

                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleOpenRunModal(proj);
                    }}
                    title="Initiate research run for this project"
                  >
                    <Play size={14} />
                    <span>New Run</span>
                  </button>

                  {!isBenchmark && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-icon"
                      onClick={(e) => {
                        e.stopPropagation();
                        setProjectToDelete(proj);
                        setIsDeleteModalOpen(true);
                      }}
                      title={`Delete project "${proj.name}"`}
                      aria-label={`Delete project ${proj.name}`}
                      style={{ color: 'var(--rose-accent)' }}
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            );
          })}

          {/* Quick Create Project Card */}
          <div
            className="project-card-empty-placeholder"
            onClick={() => setIsCreateModalOpen(true)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                setIsCreateModalOpen(true);
              }
            }}
          >
            <div className="empty-placeholder-icon">
              <FolderPlus size={24} />
            </div>
            <h3 className="empty-placeholder-title">Create Another Project</h3>
            <p className="empty-placeholder-desc">
              Define a new research domain, set baseline cutoffs, and evaluate empirical hypotheses.
            </p>
            <span className="btn btn-secondary btn-sm" style={{ marginTop: '0.5rem' }}>
              + Add Research Project
            </span>
          </div>
        </div>
      </div>

      {/* Recent Persisted Research Activity Feed */}
      <div className="u-stack-md" style={{ marginTop: '1rem' }}>
        <div className="section-header-row">
          <div>
            <h2 className="section-title">Recent Activity in the Active Project</h2>
            <p className="section-subtitle">
              Auditable activity stream derived directly from persisted papers, research knowledge, and opportunity records.
            </p>
          </div>
          <span className="badge badge-default" style={{ fontSize: '0.6875rem' }}>
            <CheckCircle2 size={12} className="text-success" style={{ marginRight: '0.25rem' }} />
            From persisted records · active project
          </span>
        </div>

        {recentActivity.length === 0 ? (
          <div className="card u-stack-sm" style={{ padding: '2rem', textAlign: 'center' }}>
            <p className="text-secondary">No recent research events recorded yet.</p>
          </div>
        ) : (
          <div className="activity-timeline">
            {recentActivity.map((act) => (
              <div key={act.id} className="activity-timeline-item">
                <div className="activity-timeline-marker">
                  {act.type === 'EVOLUTION' ? (
                    <GitBranch size={14} className="text-success" />
                  ) : act.type === 'OPPORTUNITY' ? (
                    <Compass size={14} className="text-primary" />
                  ) : (
                    <BookOpen size={14} className="text-cyan" />
                  )}
                </div>
                <div className="activity-timeline-content">
                  <div className="u-row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                    <span className="activity-title">{act.title}</span>
                    <span className={`badge badge-${act.badgeVariant}`} style={{ fontSize: '0.625rem' }}>
                      {act.badge}
                    </span>
                  </div>
                  <p className="activity-detail">{act.detail}</p>
                  <div className="u-row" style={{ justifyContent: 'space-between', marginTop: '0.375rem' }}>
                    <span className="activity-timestamp">
                      <Clock size={11} style={{ marginRight: '0.25rem', display: 'inline' }} />
                      {act.timestamp.split('T')[0]}
                    </span>
                    <button
                      type="button"
                      className="btn-link"
                      onClick={() => onNavigate(act.route)}
                    >
                      View Details &rarr;
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create Project Modal */}
      <CreateProjectModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onCreate={onCreateProject}
        onSelectProject={handleOpenProject}
      />

      {/* New Research Run Modal */}
      {runModalProject && (
        <NewRunModal
          isOpen={isRunModalOpen}
          onClose={() => {
            setIsRunModalOpen(false);
            setRunModalProject(null);
          }}
          project={runModalProject}
          onNavigate={onNavigate}
        />
      )}

      {/* Delete Project Modal */}
      {projectToDelete && (
        <DeleteProjectModal
          isOpen={isDeleteModalOpen}
          project={projectToDelete}
          onClose={() => {
            setIsDeleteModalOpen(false);
            setProjectToDelete(null);
          }}
          onDelete={async (id) => {
            if (onDeleteProject) {
              return onDeleteProject(id);
            }
            return defaultDeleteProject(id);
          }}
          onSuccess={(response, deleted) => {
            setSuccessNotification(
              `Project "${deleted.name}" was permanently deleted. Removed ${response.summary.runs_deleted} run(s), ${response.summary.gaps_deleted} gap(s), and ${response.summary.research_opportunities_deleted} opportunity proposal(s). Shared literature is preserved.`
            );
            void loadData();
          }}
        />
      )}
    </div>
  );
};
