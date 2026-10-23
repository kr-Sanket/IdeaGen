import React, { useState, useEffect, useRef } from 'react';
import { X, FolderPlus, Sparkles, AlertCircle } from 'lucide-react';
import type { ResearchProject } from '../../types';

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreate: (params: { name: string; topic: string; description?: string; temporalCutoff?: string }) => Promise<ResearchProject>;
  onSelectProject: (projectId: string) => void;
}

export const CreateProjectModal: React.FC<CreateProjectModalProps> = ({
  isOpen,
  onClose,
  onCreate,
  onSelectProject,
}) => {
  const [name, setName] = useState('');
  const [topic, setTopic] = useState('');
  const [description, setDescription] = useState('');
  const [cutoff, setCutoff] = useState('2023-12-31');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setName('');
      setTopic('');
      setDescription('');
      setCutoff('2023-12-31');
      setError(null);
      setSubmitting(false);
      setTimeout(() => nameInputRef.current?.focus(), 50);
    }
  }, [isOpen]);

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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    if (!name.trim()) {
      setError('Please provide a project name.');
      return;
    }
    if (!topic.trim()) {
      setError('Please provide a research topic or domain description.');
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      // The project is saved by the backend before it appears in the workspace.
      const created = await onCreate({
        name: name.trim(),
        topic: topic.trim(),
        description: description.trim(),
        temporalCutoff: cutoff || undefined,
      });
      onSelectProject(created.id);
      onClose();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create project.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="u-row">
            <div className="modal-icon-badge">
              <FolderPlus size={20} />
            </div>
            <div>
              <h2 id="modal-title" className="modal-title">Create Research Project</h2>
              <p className="modal-subtitle">Configure a new research domain within your personal workspace</p>
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

        <form onSubmit={handleSubmit} className="modal-body u-stack-md">
          {error && (
            <div className="form-error-banner" role="alert">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <div className="form-group">
            <label htmlFor="proj-name" className="form-label">
              Project Name <span className="text-danger">*</span>
            </label>
            <input
              id="proj-name"
              ref={nameInputRef}
              type="text"
              className="form-input"
              placeholder="e.g. Agentic Multi-Step Reasoning & Tool Selection"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="proj-topic" className="form-label">
              Research Topic / Hypothesis Domain <span className="text-danger">*</span>
            </label>
            <textarea
              id="proj-topic"
              className="form-textarea"
              rows={2}
              placeholder="e.g. Investigating tool-augmented LLM reasoning trajectories, self-correction limitations, and benchmark evaluation"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              aria-describedby="proj-topic-help"
              required
            />
            <span id="proj-topic-help" className="form-helper">
              Describes the project's research scope. It is stored with the project for reference.
            </span>
          </div>

          <div className="form-group">
            <label htmlFor="proj-desc" className="form-label">
              Description / Notes (Optional)
            </label>
            <textarea
              id="proj-desc"
              className="form-textarea"
              rows={2}
              placeholder="Optional methodological notes or experimental focus"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label htmlFor="proj-cutoff" className="form-label">
              Baseline Temporal Cutoff Date
            </label>
            <input
              id="proj-cutoff"
              type="date"
              className="form-input"
              value={cutoff}
              onChange={(e) => setCutoff(e.target.value)}
              aria-describedby="proj-cutoff-help"
            />
            <span id="proj-cutoff-help" className="form-helper">
              Literature published on or before this cutoff forms the foundational knowledge base; subsequent literature serves as temporal evaluation data.
            </span>
          </div>

          <div className="callout callout-info">
            <div className="callout-icon-col">
              <Sparkles size={16} />
            </div>
            <div className="callout-content">
              <div className="callout-title">Project Isolation Notice</div>
              <p className="callout-body">
                The project is saved by the backend. Papers live in one shared literature store and are linked to each project that adds them; gaps, opportunities and research runs belong to a single project.
              </p>
            </div>
          </div>

          <div className="modal-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting} aria-busy={submitting}>
              <FolderPlus size={16} aria-hidden="true" />
              <span>{submitting ? 'Creating…' : 'Create Project'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
