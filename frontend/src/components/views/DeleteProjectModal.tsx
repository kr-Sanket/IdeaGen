import React, { useState, useEffect, useRef } from 'react';
import { X, Trash2, AlertTriangle, ShieldAlert, RefreshCw, CheckCircle2 } from 'lucide-react';
import type { ResearchProject, ProjectDeleteResponse } from '../../types';

interface DeleteProjectModalProps {
  isOpen: boolean;
  project: ResearchProject | null;
  onClose: () => void;
  onDelete: (projectId: string) => Promise<ProjectDeleteResponse>;
  onSuccess: (response: ProjectDeleteResponse, deletedProject: ResearchProject) => void;
}

export const DeleteProjectModal: React.FC<DeleteProjectModalProps> = ({
  isOpen,
  project,
  onClose,
  onDelete,
  onSuccess,
}) => {
  const [confirmationInput, setConfirmationInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const isBenchmark = Boolean(
    project && (project.id === 'rag-benchmark' || project.isDefaultBenchmark)
  );

  useEffect(() => {
    if (isOpen) {
      setConfirmationInput('');
      setError(null);
      setSubmitting(false);
      if (!isBenchmark) {
        setTimeout(() => inputRef.current?.focus(), 60);
      }
    }
  }, [isOpen, project, isBenchmark]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !submitting) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, submitting, onClose]);

  if (!isOpen || !project) return null;

  const isConfirmed = confirmationInput.trim() === project.name.trim();

  const handleConfirmDelete = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isBenchmark || !isConfirmed || submitting) return;

    setSubmitting(true);
    setError(null);

    try {
      const response = await onDelete(project.id);
      onSuccess(response, project);
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to delete project. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="modal-backdrop"
      onClick={submitting ? undefined : onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-modal-title"
    >
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '540px' }}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="u-row" style={{ gap: '0.75rem' }}>
            <div
              className="modal-icon-badge"
              style={{
                background: isBenchmark ? 'var(--amber-light, #fef3c7)' : 'var(--rose-light, #ffe4e6)',
                color: isBenchmark ? 'var(--amber-accent, #d97706)' : 'var(--rose-accent, #e11d48)',
              }}
            >
              {isBenchmark ? <ShieldAlert size={20} /> : <Trash2 size={20} />}
            </div>
            <div>
              <h2 id="delete-modal-title" className="modal-title">
                {isBenchmark ? 'Protected Project' : 'Delete Research Project'}
              </h2>
              <p className="modal-subtitle">
                {isBenchmark
                  ? 'System-level primary benchmark project'
                  : 'Permanently remove project and all owned research artifacts'}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            disabled={submitting}
            aria-label="Close dialog"
          >
            <X size={18} />
          </button>
        </div>

        {/* Modal Body */}
        {isBenchmark ? (
          <div className="modal-body u-stack-md">
            <div className="callout callout-warning">
              <div className="callout-icon-col">
                <ShieldAlert size={18} />
              </div>
              <div className="callout-content">
                <div className="callout-title">Protected Benchmark Project</div>
                <p className="callout-body">
                  <strong>{project.name}</strong> is the primary benchmark baseline for IdeaGen. It cannot be deleted or deregistered from the workspace or server.
                </p>
              </div>
            </div>
            <div className="modal-actions">
              <button type="button" className="btn btn-secondary" onClick={onClose}>
                Close
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={handleConfirmDelete} className="modal-body u-stack-md">
            {error && (
              <div className="form-error-banner" role="alert">
                <AlertTriangle size={16} />
                <span>{error}</span>
              </div>
            )}

            {/* Target Project Summary Box */}
            <div
              className="card"
              style={{
                padding: '0.875rem 1rem',
                background: 'var(--bg-subtle)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div className="label-caps text-tertiary" style={{ fontSize: '0.6875rem', marginBottom: '0.25rem' }}>
                Project to be deleted
              </div>
              <div className="font-bold" style={{ fontSize: '1rem', color: 'var(--text-primary)' }}>
                {project.name}
              </div>
              <div className="text-secondary" style={{ fontSize: '0.8125rem', marginTop: '0.25rem' }}>
                {project.topic}
              </div>
            </div>

            {/* Scope of deletion & retention */}
            <div
              style={{
                display: 'grid',
                gap: '0.75rem',
                fontSize: '0.8125rem',
                lineHeight: '1.45',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  gap: '0.625rem',
                  padding: '0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--rose-light, #ffe4e6)',
                  border: '1px solid var(--rose-border, #fecdd3)',
                  color: 'var(--text-primary)',
                }}
              >
                <Trash2 size={16} style={{ color: 'var(--rose-accent)', flexShrink: 0, marginTop: '0.125rem' }} />
                <div>
                  <strong style={{ color: 'var(--rose-accent)' }}>Permanently removed:</strong>
                  <div style={{ color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    All project-owned research runs, logs, run papers, paper selections, discovered gaps, gap evidence links, research opportunities, prior art assessments, refinement iterations, lifecycle opportunities, and transition history.
                  </div>
                </div>
              </div>

              <div
                style={{
                  display: 'flex',
                  gap: '0.625rem',
                  padding: '0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--emerald-light, rgba(16, 185, 129, 0.08))',
                  border: '1px solid var(--emerald-border, rgba(16, 185, 129, 0.25))',
                  color: 'var(--text-primary)',
                }}
              >
                <CheckCircle2 size={16} style={{ color: 'var(--emerald-accent, #059669)', flexShrink: 0, marginTop: '0.125rem' }} />
                <div>
                  <strong style={{ color: 'var(--emerald-accent, #059669)' }}>Safely retained:</strong>
                  <div style={{ color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                    Shared literature, extracted statements, passages, and underlying paper records in the global database remain intact and accessible to other projects.
                  </div>
                </div>
              </div>
            </div>

            {/* Deliberate Confirmation Input */}
            <div className="form-group" style={{ marginTop: '0.25rem' }}>
              <label htmlFor="confirm-project-name" className="form-label">
                To confirm permanent deletion, type the project name{' '}
                <strong style={{ userSelect: 'all', color: 'var(--text-primary)' }}>"{project.name}"</strong>:
              </label>
              <input
                id="confirm-project-name"
                ref={inputRef}
                type="text"
                className="form-input"
                autoComplete="off"
                placeholder={project.name}
                value={confirmationInput}
                onChange={(e) => setConfirmationInput(e.target.value)}
                disabled={submitting}
                aria-describedby="confirm-help"
                required
              />
              <span id="confirm-help" className="form-helper">
                This action cannot be undone. Enter the exact project name above.
              </span>
            </div>

            {/* Modal Actions */}
            <div className="modal-actions" style={{ marginTop: '0.5rem' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={onClose}
                disabled={submitting}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-danger"
                disabled={!isConfirmed || submitting}
                aria-busy={submitting}
                id="btn-confirm-delete-project"
              >
                {submitting ? (
                  <>
                    <RefreshCw size={14} className="animate-spin" aria-hidden="true" />
                    <span>Deleting Project…</span>
                  </>
                ) : (
                  <>
                    <Trash2 size={14} aria-hidden="true" />
                    <span>Permanently Delete Project</span>
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
