import React, { useCallback, useEffect, useState } from 'react';
import {
  Layers,
  BookOpen,
  FileText,
  Lightbulb,
  Compass,
  ArrowRight,
  Calendar,
  Cpu,
  RefreshCw,
} from 'lucide-react';
import type { ResearchProject, WorkspaceStats, SystemStatus } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { ErrorBanner } from '../common/ErrorBanner';

interface ProjectOverviewViewProps {
  project: ResearchProject;
  onNavigate: (route: string) => void;
}

function describeModels(status: SystemStatus | null): string {
  if (!status) return 'Unavailable';
  const llm = status.llm.model ? `${status.llm.provider} ${status.llm.model}` : status.llm.provider;
  const emb = status.embedding.model
    ? `${status.embedding.model}${status.embedding.dimension ? ` (${status.embedding.dimension}-dim)` : ''}`
    : status.embedding.provider;
  return `${llm} · ${emb}`;
}

function describeProviderHealth(status: SystemStatus | null): { label: string; ok: boolean } {
  if (!status) return { label: 'Unknown (status unavailable)', ok: false };
  if (!status.database.connected) return { label: 'Database error', ok: false };
  const ollama = status.ollama;
  if (ollama && !ollama.reachable) return { label: 'Ollama unreachable', ok: false };
  if (ollama && ollama.missing_models.length > 0) return { label: `Missing model: ${ollama.missing_models.join(', ')}`, ok: false };
  return { label: 'Database and models available', ok: true };
}

export const ProjectOverviewView: React.FC<ProjectOverviewViewProps> = ({ project, onNavigate }) => {
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const [statsResult, systemResult] = await Promise.allSettled([
      api.getWorkspaceStats(project.id),
      api.getSystemStatus(),
    ]);
    if (statsResult.status === 'fulfilled') {
      setStats(statsResult.value);
    } else {
      setStats(null);
      setError(statsResult.reason instanceof Error ? statsResult.reason.message : 'Failed to load project metrics.');
    }
    setSystem(systemResult.status === 'fulfilled' ? systemResult.value : null);
    setLoading(false);
  }, [project.id]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  if (loading) {
    return <LoadingState message="Loading project overview & metrics..." minHeight="400px" />;
  }

  const isBenchmark = project.isDefaultBenchmark;
  const health = describeProviderHealth(system);
  // Every count below is scoped to this project (GET /api/v1/stats?project_id=...).
  const metric = (value: number | undefined) => (stats ? String(value ?? 0) : '—');

  return (
    <div className="view-container animate-fade-in">
      {/* Project Overview Header */}
      <div className="view-header">
        <div>
          <div className="u-row" style={{ gap: '0.625rem', marginBottom: '0.25rem' }}>
            {isBenchmark ? (
              <span className="badge badge-primary">Primary Benchmark</span>
            ) : (
              <span className="badge badge-default">Custom Project</span>
            )}
            <span className="badge badge-success">{project.status}</span>
          </div>
          <h1 className="view-title">{project.name}</h1>
          <p className="view-subtitle">{project.topic}</p>
        </div>
        <div className="u-row" style={{ gap: '0.75rem' }}>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => void loadData()}
            title="Refresh overview metrics"
          >
            <RefreshCw size={14} aria-hidden="true" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <ErrorBanner title="Could Not Load Project Metrics" message={error} onRetry={() => void loadData()} />
      )}

      {/* Project Context & Temporal Cutoff Bar */}
      <div className="card" style={{ padding: '1rem 1.25rem' }}>
        <div className="u-row" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div className="u-row" style={{ gap: '0.75rem' }}>
            <Calendar size={18} className="text-primary" aria-hidden="true" />
            <div>
              <span className="label-caps text-tertiary">Baseline Temporal Cutoff</span>
              <div className="font-semibold">{project.temporalCutoff || 'No cutoff specified'}</div>
            </div>
          </div>

          <div className="u-row" style={{ gap: '0.75rem' }}>
            <Cpu size={18} className="text-cyan" aria-hidden="true" />
            <div>
              <span className="label-caps text-tertiary">Configured Models</span>
              <div className="font-semibold">{describeModels(system)}</div>
            </div>
          </div>

          <div className="u-row" style={{ gap: '0.75rem' }}>
            <Layers size={18} className="text-emerald" aria-hidden="true" />
            <div>
              <span className="label-caps text-tertiary">Provider Status</span>
              <div className={`font-semibold ${health.ok ? 'text-success' : 'text-danger'}`}>{health.label}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Metric Cards Grid (project-scoped) */}
      <div className="stat-cards-grid">
        <button
          type="button"
          className="stat-card u-clickable"
          onClick={() => onNavigate(`#/projects/${project.id}/papers?tab=corpus`)}
        >
          <div className="stat-card-header">
            <span className="stat-card-label">Corpus Papers</span>
            <span className="stat-card-icon stat-icon-primary">
              <BookOpen size={18} aria-hidden="true" />
            </span>
          </div>
          <div className="stat-card-value">{metric(stats?.total_papers)}</div>
          <div className="stat-card-sub">
            {stats ? `${stats.analyzed_papers} with extracted statements` : 'Not loaded'}
          </div>
        </button>

        <button
          type="button"
          className="stat-card u-clickable"
          onClick={() => onNavigate(`#/projects/${project.id}/papers?tab=corpus`)}
        >
          <div className="stat-card-header">
            <span className="stat-card-label">Research Statements</span>
            <span className="stat-card-icon stat-icon-cyan">
              <FileText size={18} aria-hidden="true" />
            </span>
          </div>
          <div className="stat-card-value">{metric(stats?.total_statements)}</div>
          <div className="stat-card-sub">
            {stats ? `from ${stats.total_passages} stored passages` : 'Not loaded'}
          </div>
        </button>

        <button
          type="button"
          className="stat-card u-clickable"
          onClick={() => onNavigate(`#/projects/${project.id}/gaps`)}
        >
          <div className="stat-card-header">
            <span className="stat-card-label">Research Gaps</span>
            <span className="stat-card-icon stat-icon-amber">
              <Lightbulb size={18} aria-hidden="true" />
            </span>
          </div>
          <div className="stat-card-value">{metric(stats?.total_gaps)}</div>
          <div className="stat-card-sub">
            {stats
              ? `${stats.validated_gaps} validated · ${stats.candidate_gaps} candidate · ${stats.rejected_gaps} rejected`
              : 'Not loaded'}
          </div>
        </button>

        <button
          type="button"
          className="stat-card u-clickable"
          onClick={() => onNavigate(`#/projects/${project.id}/opportunities`)}
        >
          <div className="stat-card-header">
            <span className="stat-card-label">Generated Opportunities</span>
            <span className="stat-card-icon stat-icon-emerald">
              <Compass size={18} aria-hidden="true" />
            </span>
          </div>
          <div className="stat-card-value">{metric(stats?.total_opportunities)}</div>
          <div className="stat-card-sub">
            {stats ? `${stats.validated_opportunities} validated or refined` : 'Not loaded'}
          </div>
        </button>
      </div>

      {/* Pipeline Navigation / Stages Progress */}
      <div className="u-stack-md">
        <h2 className="section-title">Research Pipeline Stages</h2>
        <div className="workflow-steps-list">
          <div className="workflow-step-card">
            <div className="workflow-step-num">1</div>
            <div className="workflow-step-info">
              <h3 className="workflow-step-title">Literature Ingestion (OpenAlex)</h3>
              <p className="workflow-step-desc">
                Search OpenAlex, harvest open-access PDFs, extract clean body text, and chunk into passages.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onNavigate(`#/projects/${project.id}/papers?tab=search`)}
            >
              <span>View Papers</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="workflow-step-card">
            <div className="workflow-step-num">2</div>
            <div className="workflow-step-info">
              <h3 className="workflow-step-title">Statement Extraction &amp; Grounding</h3>
              <p className="workflow-step-desc">
                Extract verifiable claims, findings, and limitation statements linked to source passages.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onNavigate(`#/projects/${project.id}/papers?tab=corpus`)}
            >
              <span>Inspect Statements</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="workflow-step-card">
            <div className="workflow-step-num">3</div>
            <div className="workflow-step-info">
              <h3 className="workflow-step-title">Research Gap Discovery</h3>
              <p className="workflow-step-desc">
                Identify unaddressed technical challenges and cluster contradicting or unvalidated statements.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onNavigate(`#/projects/${project.id}/gaps`)}
            >
              <span>Explore Gaps</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="workflow-step-card">
            <div className="workflow-step-num">4</div>
            <div className="workflow-step-info">
              <h3 className="workflow-step-title">Structured Opportunity Formulation</h3>
              <p className="workflow-step-desc">
                After reviewing a validated gap, explicitly generate a 16-section opportunity proposal. Runs never do this automatically.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onNavigate(`#/projects/${project.id}/opportunities`)}
            >
              <span>Explore Proposals</span>
              <ArrowRight size={13} />
            </button>
          </div>

          <div className="workflow-step-card">
            <div className="workflow-step-num">5</div>
            <div className="workflow-step-info">
              <h3 className="workflow-step-title">Longitudinal Evolution Tracking</h3>
              <p className="workflow-step-desc">
                Classify subsequent literature against opportunities to track transition from OPEN to RESOLVED.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onNavigate(`#/projects/${project.id}/evolution`)}
            >
              <span>Track Evolution</span>
              <ArrowRight size={13} />
            </button>
          </div>
        </div>
      </div>

    </div>
  );
};
