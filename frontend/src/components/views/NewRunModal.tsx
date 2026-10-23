import React, { useEffect, useState } from 'react';
import { X, Play, BookOpen, FileSearch, Lightbulb, Compass, GitBranch, ArrowRight, Layers } from 'lucide-react';
import type { ResearchProject, WorkspaceStats } from '../../types';
import { api } from '../../api/client';
import { RunPreflightModal } from '../workflow/RunPreflightModal';
import { RunProgressModal } from '../workflow/RunProgressModal';

interface NewRunModalProps {
  isOpen: boolean;
  onClose: () => void;
  project: ResearchProject;
  onNavigate: (route: string) => void;
}

export const NewRunModal: React.FC<NewRunModalProps> = ({
  isOpen,
  onClose,
  project,
  onNavigate,
}) => {
  const [isPreflightOpen, setIsPreflightOpen] = useState<boolean>(false);
  const [isProgressOpen, setIsProgressOpen] = useState<boolean>(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [projectStats, setProjectStats] = useState<WorkspaceStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);

  // Stage readiness is derived from this project's own counts, loaded whenever the modal opens.
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    api
      .getWorkspaceStats(project.id)
      .then((s) => { if (!cancelled) { setProjectStats(s); setStatsError(null); } })
      .catch((err: unknown) => {
        if (!cancelled) {
          setProjectStats(null);
          setStatsError(err instanceof Error ? err.message : 'Could not load project counts.');
        }
      });
    return () => { cancelled = true; };
  }, [isOpen, project.id]);
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleStartWorkflow = (stepRoute: string) => {
    onClose();
    onNavigate(stepRoute);
  };

  const stats = {
    papers: projectStats?.total_papers ?? 0,
    statements: projectStats?.total_statements ?? 0,
    gaps: projectStats?.total_gaps ?? 0,
    validatedGaps: projectStats?.validated_gaps ?? 0,
    opportunities: projectStats?.total_opportunities ?? 0,
  };

  const steps = [
    {
      num: 1,
      title: 'Literature Search & Ingestion',
      desc: 'Query OpenAlex for domain literature, download open-access PDFs, and extract structured passages.',
      route: `#/projects/${project.id}/papers?tab=search`,
      countLabel: `${stats.papers} papers stored`,
      icon: <BookOpen size={18} />,
      status: stats.papers > 0 ? 'READY' : 'REQUIRED',
    },
    {
      num: 2,
      title: 'Statement Extraction & Understanding',
      desc: 'Run the configured LLM over stored passages to extract evidenced research statements.',
      route: `#/projects/${project.id}/papers?tab=corpus`,
      countLabel: `${stats.statements} statements extracted`,
      icon: <FileSearch size={18} />,
      status: stats.statements > 0 ? 'READY' : (stats.papers > 0 ? 'NEXT' : 'BLOCKED'),
    },
    {
      num: 3,
      title: 'Hypothesis & Gap Discovery',
      desc: 'Cluster extracted statements and identify unresolved contradictions or unaddressed limitations.',
      route: `#/projects/${project.id}/gaps`,
      countLabel: `${stats.gaps} gaps identified`,
      icon: <Lightbulb size={18} />,
      status: stats.gaps > 0 ? 'READY' : (stats.statements > 0 ? 'NEXT' : 'BLOCKED'),
    },
    {
      num: 4,
      title: 'Structured Opportunity Formulation',
      desc: 'After reviewing a validated gap, explicitly generate a 16-section research proposal from it. Runs never do this automatically.',
      route: `#/projects/${project.id}/gaps`,
      countLabel: `${stats.opportunities} opportunities generated`,
      icon: <Compass size={18} />,
      status: stats.opportunities > 0 ? 'READY' : (stats.validatedGaps > 0 ? 'NEXT' : 'BLOCKED'),
    },
    {
      num: 5,
      title: 'Temporal Evolution & Validation',
      desc: 'Track how subsequent papers published after the cutoff address, contradict, or resolve opportunities.',
      route: `#/projects/${project.id}/evolution`,
      countLabel: 'Opportunities available to evolve',
      icon: <GitBranch size={18} />,
      status: stats.opportunities > 0 ? 'READY' : 'BLOCKED',
    },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="run-modal-title">
      <div className="modal-dialog modal-dialog-lg" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="u-row">
            <div className="modal-icon-badge modal-icon-badge-primary">
              <Play size={20} />
            </div>
            <div>
              <h2 id="run-modal-title" className="modal-title">Initiate Research Run</h2>
              <p className="modal-subtitle">
                Project: <strong className="text-primary">{project.name}</strong> · Cutoff: {project.temporalCutoff || 'None'}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body u-stack-md">
          {statsError && <p className="inline-note is-error" role="alert">{statsError}</p>}

          <div className="nav-callout-card">
            <div>
              <div className="nav-callout-title">Automated Research Run</div>
              <p className="nav-callout-desc">
                Review the project's corpus and any queued papers, then run ingestion, statement extraction and gap discovery.
                The run pauses for your review; opportunities are only generated when you request one from a validated gap.
              </p>
            </div>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => setIsPreflightOpen(true)}
            >
              <Layers size={14} aria-hidden="true" />
              <span>Preflight &amp; Run</span>
            </button>
          </div>

          <p className="text-secondary" style={{ fontSize: '0.875rem' }}>
            Or jump directly to an individual workflow stage below:
          </p>

          <div className="workflow-steps-list">
            {steps.map((s) => (
              <div key={s.num} className={`workflow-step-card ${s.status === 'NEXT' ? 'is-recommended' : ''}`}>
                <div className="workflow-step-num">{s.num}</div>
                <div className="workflow-step-info">
                  <div className="workflow-step-header">
                    <span className="workflow-step-icon">{s.icon}</span>
                    <h3 className="workflow-step-title">{s.title}</h3>
                    {s.status === 'NEXT' && (
                      <span className="badge badge-success" style={{ fontSize: '0.6875rem' }}>
                        Recommended Next
                      </span>
                    )}
                    {s.status === 'READY' && (
                      <span className="badge badge-default" style={{ fontSize: '0.6875rem' }}>
                        {s.countLabel}
                      </span>
                    )}
                  </div>
                  <p className="workflow-step-desc">{s.desc}</p>
                </div>
                <button
                  type="button"
                  className={`btn ${s.status === 'NEXT' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => handleStartWorkflow(s.route)}
                >
                  <span>Go to Stage</span>
                  <ArrowRight size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>

      <RunPreflightModal
        isOpen={isPreflightOpen}
        onClose={() => setIsPreflightOpen(false)}
        projectId={project.id}
        projectName={project.name}
        onStartRun={(runId) => {
          setActiveRunId(runId);
          setIsProgressOpen(true);
        }}
      />

      <RunProgressModal
        isOpen={isProgressOpen}
        onClose={() => setIsProgressOpen(false)}
        runId={activeRunId}
        projectId={project.id}
        onNavigate={onNavigate}
      />
    </div>
  );
};

