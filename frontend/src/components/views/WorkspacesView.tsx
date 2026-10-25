import React, { useEffect, useState } from 'react';
import { Shield, Lock, ArrowRight, UserCheck, HardDrive } from 'lucide-react';
import type { ResearchProject, SystemStatus } from '../../types';
import { api } from '../../api/client';

interface WorkspacesViewProps {
  projects: ResearchProject[];
  activeProjectId: string;
  onSelectProject: (projectId: string) => void;
  onNavigate: (route: string) => void;
}

export const WorkspacesView: React.FC<WorkspacesViewProps> = ({
  projects,
  activeProjectId,
  onSelectProject,
  onNavigate,
}) => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getSystemStatus()
      .then((s) => { if (!cancelled) setStatus(s); })
      .catch((err: unknown) => { if (!cancelled) setStatusError(err instanceof Error ? err.message : 'Status unavailable'); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="view-container animate-fade-in">
      <div className="view-header">
        <div>
          <div className="u-row" style={{ gap: '0.625rem', marginBottom: '0.25rem' }}>
            <span className="badge badge-primary">Personal Workspace</span>
            <span className="badge badge-default">Single Owner</span>
          </div>
          <h1 className="view-title">Research Workspaces</h1>
          <p className="view-subtitle">
            Personal workspace architecture and data governance for empirical literature analysis and opportunity generation.
          </p>
        </div>
      </div>

      <div className="grid-2col">
        {/* Workspace Identity Card */}
        <div className="card u-stack-md">
          <div className="u-row" style={{ gap: '0.75rem' }}>
            <div className="modal-icon-badge modal-icon-badge-primary">
              <UserCheck size={20} />
            </div>
            <div>
              <h2 className="card-title">Personal Workspace</h2>
              <span className="text-secondary" style={{ fontSize: '0.8125rem' }}>
                Primary Private Research Environment
              </span>
            </div>
          </div>

          <p className="text-secondary" style={{ fontSize: '0.875rem', lineHeight: '1.6' }}>
            This workspace houses all your empirical research projects, ingested literature collections, extracted statement evidence, discovered research gaps, and longitudinal opportunity evolution history.
          </p>

          <div className="callout callout-info">
            <div className="callout-icon-col">
              <Lock size={16} />
            </div>
            <div className="callout-content">
              <div className="callout-title">Private by Design · Sharing Deferred</div>
              <p className="callout-body">
                Projects and research trajectories in this workspace are strictly private to the owner. Cross-user collaboration, organization-level tenancy, and public project sharing are deferred to subsequent releases.
              </p>
            </div>
          </div>

          <div className="u-stack-sm" style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1rem' }}>
            <div className="u-row" style={{ justifyContent: 'space-between', fontSize: '0.875rem' }}>
              <span className="text-secondary">Workspace Status:</span>
              <span className="text-success font-semibold">Active &amp; Operational</span>
            </div>
            <div className="u-row" style={{ justifyContent: 'space-between', fontSize: '0.875rem' }}>
              <span className="text-secondary">Registered Projects:</span>
              <span className="font-semibold">{projects.length} project{projects.length !== 1 ? 's' : ''}</span>
            </div>
            <div className="u-row" style={{ justifyContent: 'space-between', fontSize: '0.875rem' }}>
              <span className="text-secondary">Active Context:</span>
              <span className="text-primary font-semibold">
                {projects.find((p) => p.id === activeProjectId)?.name || 'Default Project'}
              </span>
            </div>
          </div>
        </div>

        {/* Storage & Environment Information */}
        <div className="card u-stack-md">
          <div className="u-row" style={{ gap: '0.75rem' }}>
            <div className="modal-icon-badge">
              <HardDrive size={20} />
            </div>
            <div>
              <h2 className="card-title">Database &amp; Storage Subsystem</h2>
              <span className="text-secondary" style={{ fontSize: '0.8125rem' }}>
                Local Persistence Architecture
              </span>
            </div>
          </div>

          <div className="u-stack-sm">
            <div className="detail-row">
              <span className="detail-label">Database Engine</span>
              <span className="detail-value">
                {status
                  ? `${status.database.dialect ?? 'unknown'} · ${status.database.connected ? 'connected' : 'connection error'}`
                  : statusError ?? 'Checking…'}
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Archived Baselines</span>
              <span className="detail-value text-secondary">
                experiments/artifacts/ (Read-only, untouched)
              </span>
            </div>

            <div className="detail-row">
              <span className="detail-label">Vector Embeddings</span>
              <span className="detail-value">
                {status?.embedding.model
                  ? `${status.embedding.model}${status.embedding.dimension ? ` (${status.embedding.dimension}-dim)` : ''} · ${status.embedding.provider}`
                  : status ? status.embedding.provider : '—'}
              </span>
            </div>
          </div>

          <div className="callout callout-warning">
            <div className="callout-icon-col">
              <Shield size={16} />
            </div>
            <div className="callout-content">
              <div className="callout-title">Project Data Scoping</div>
              <p className="callout-body">
                Papers and passages live in one shared literature store and are linked to projects through
                `project_papers`; gaps, opportunities and runs carry a `project_id`. This is a single-user workspace,
                not multi-tenant: there is no authentication.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Projects in this Workspace */}
      <div className="u-stack-md" style={{ marginTop: '1rem' }}>
        <h2 className="section-title">Projects in this Workspace</h2>
        <div className="card-table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Project Name</th>
                <th>Research Topic</th>
                <th>Baseline Cutoff</th>
                <th>Status</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((proj) => {
                const isActive = proj.id === activeProjectId;
                return (
                  <tr key={proj.id} className={isActive ? 'table-row-active' : ''}>
                    <td>
                      <div className="u-stack-xs">
                        <span className="font-semibold text-primary">{proj.name}</span>
                        {proj.isDefaultBenchmark && (
                          <span className="badge badge-primary" style={{ fontSize: '0.5625rem', width: 'fit-content' }}>
                            Benchmark
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className="text-secondary" style={{ fontSize: '0.8125rem', maxWidth: '400px', display: 'inline-block' }}>
                        {proj.topic}
                      </span>
                    </td>
                    <td>
                      <span className="font-mono text-tertiary" style={{ fontSize: '0.8125rem' }}>
                        {proj.temporalCutoff || '—'}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${isActive ? 'badge-success' : 'badge-default'}`}>
                        {isActive ? 'Active' : proj.status}
                      </span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => {
                          onSelectProject(proj.id);
                          onNavigate(`#/projects/${proj.id}/overview`);
                        }}
                      >
                        <span>Open</span>
                        <ArrowRight size={13} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
