import React, { useEffect, useState } from 'react';
import {
  X,
  Play,
  AlertTriangle,
  BookOpen,
  Layers,
  Sparkles,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { api } from '../../api/client';
import type { PreflightResponse } from '../../types';
import { Badge } from '../common/Badge';
import type { BadgeVariant } from '../common/badgeVariants';

interface RunPreflightModalProps {
  isOpen: boolean;
  onClose: () => void;
  projectId: string;
  projectName?: string;
  onStartRun: (runId: string) => void;
}

export const RunPreflightModal: React.FC<RunPreflightModalProps> = ({
  isOpen,
  onClose,
  projectId,
  projectName = 'Research Project',
  onStartRun,
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PreflightResponse | null>(null);
  const [submitting, setSubmitting] = useState<boolean>(false);

  const fetchPreflight = () => {
    if (!projectId) return;
    setLoading(true);
    setError(null);
    api
      .getPreflightCheck(projectId)
      .then((res) => {
        setData(res);
      })
      .catch((err: Error) => {
        setError(err.message || 'Failed to load preflight analysis.');
      })
      .finally(() => {
        setLoading(false);
      });
  };

  useEffect(() => {
    if (isOpen) {
      fetchPreflight();
    }
  }, [isOpen, projectId]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !submitting) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, submitting, onClose]);

  if (!isOpen) return null;

  const handleConfirmRun = async () => {
    if (!projectId || submitting) return;
    setSubmitting(true);
    try {
      const run = await api.createResearchRun(projectId);
      onStartRun(run.id);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to initiate research run.');
      setSubmitting(false);
    }
  };

  const getStatusBadgeVariant = (status: string): BadgeVariant => {
    switch (status) {
      case 'CORPUS_MATCH':
        return 'status-positive';
      case 'NO_PDF':
      case 'METADATA_ONLY':
        return 'status-warning';
      default:
        return 'neutral';
    }
  };

  const formatStatusLabel = (status: string) => {
    switch (status) {
      case 'CORPUS_MATCH':
        return 'In Corpus (Reused)';
      case 'NEW':
        return 'New to Ingest';
      case 'NO_PDF':
        return 'No PDF (Metadata)';
      case 'METADATA_ONLY':
        return 'Metadata Only';
      default:
        return status;
    }
  };

  const totalSelected = data ? data.total_selected : 0;
  const corpusMatches = data ? data.corpus_matches : 0;
  const newPapers = data ? data.new_papers : 0;
  const pdfAvailable = data ? data.pdf_available : 0;
  const metadataOnly = data ? data.metadata_only : 0;
  const corpus = data?.corpus ?? null;
  const corpusTotal = corpus?.total ?? 0;
  // The run processes queued selections and the project's existing corpus.
  const runnable = data ? (data.runnable ?? totalSelected > 0) : false;

  return (
    <div
      className="modal-backdrop"
      onClick={() => {
        if (!submitting) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="preflight-modal-title"
    >
      <div
        className="modal-dialog modal-dialog-lg preflight-modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '820px' }}
      >
        <div className="modal-header">
          <div className="u-row" style={{ gap: '0.875rem' }}>
            <div className="modal-icon-badge modal-icon-badge-primary">
              <Layers size={20} />
            </div>
            <div>
              <h2 id="preflight-modal-title" className="modal-title">
                Research Run Preflight Review
              </h2>
              <p className="modal-subtitle">
                Project: <strong style={{ color: 'var(--text-secondary)' }}>{projectName}</strong> · Stage verification &amp; duplicate detection
              </p>
            </div>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onClose}
            disabled={submitting}
            aria-label="Close preflight review"
          >
            <X size={18} />
          </button>
        </div>

        <div className="modal-body u-stack" style={{ gap: '1.25rem' }}>
          {loading && (
            <div className="inline-pending" style={{ padding: '3rem 1rem', justifyContent: 'center' }}>
              <Loader2 size={24} className="spin" />
              <span>Analyzing selected papers, corpus matches, and execution stages…</span>
            </div>
          )}

          {error && !loading && (
            <div className="action-result is-error" role="alert">
              <p className="action-result-title">Preflight Check Error</p>
              <p className="u-text-sm">{error}</p>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={fetchPreflight}
                style={{ marginTop: '0.5rem', alignSelf: 'flex-start' }}
              >
                <RefreshCw size={12} />
                <span>Retry Analysis</span>
              </button>
            </div>
          )}

          {!loading && data && (
            <>
              {/* Metrics Summary Strip */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                  gap: '0.75rem',
                }}
              >
                <div className="card" style={{ padding: '0.875rem 1rem', background: 'rgba(255, 255, 255, 0.02)' }}>
                  <span className="u-text-xs u-text-tertiary">Queued Papers</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {totalSelected}
                  </div>
                </div>

                <div className="card" style={{ padding: '0.875rem 1rem', background: 'rgba(16, 185, 129, 0.05)' }}>
                  <span className="u-text-xs" style={{ color: '#10b981' }}>Queued, Already Stored</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10b981' }}>
                    {corpusMatches}
                  </div>
                </div>

                <div className="card" style={{ padding: '0.875rem 1rem', background: 'rgba(59, 130, 246, 0.05)' }}>
                  <span className="u-text-xs" style={{ color: '#3b82f6' }}>New to Ingest</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#3b82f6' }}>
                    {newPapers}
                  </div>
                </div>

                <div className="card" style={{ padding: '0.875rem 1rem', background: 'rgba(99, 102, 241, 0.05)' }}>
                  <span className="u-text-xs" style={{ color: 'var(--primary)' }}>PDF Available</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)' }}>
                    {pdfAvailable}
                  </div>
                </div>

                <div className="card" style={{ padding: '0.875rem 1rem', background: 'rgba(245, 158, 11, 0.05)' }}>
                  <span className="u-text-xs" style={{ color: '#f59e0b' }}>Metadata Only</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f59e0b' }}>
                    {metadataOnly}
                  </div>
                </div>
              </div>

              {/* Warnings Box */}
              {data.warnings && data.warnings.length > 0 && (
                <div
                  style={{
                    padding: '0.875rem 1rem',
                    borderRadius: 'var(--radius-md)',
                    background: 'rgba(245, 158, 11, 0.08)',
                    border: '1px solid rgba(245, 158, 11, 0.25)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.5rem',
                  }}
                  role="status"
                >
                  <div className="u-row" style={{ gap: '0.5rem', color: '#f59e0b', fontWeight: 600, fontSize: '0.8125rem' }}>
                    <AlertTriangle size={15} />
                    <span>Run Preflight Notices ({data.warnings.length})</span>
                  </div>
                  <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {data.warnings.map((w, i) => (
                      <li key={i} style={{ marginBottom: '0.25rem' }}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Planned Stages */}
              <div className="u-stack" style={{ gap: '0.5rem' }}>
                <span className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Sparkles size={14} style={{ color: 'var(--primary)' }} />
                  <span>Planned Sequential Pipeline Stages</span>
                </span>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '0.5rem',
                  }}
                >
                  {data.planned_stages.map((st, idx) => (
                    <div
                      key={st.stage_id || st.stage || idx}
                      style={{
                        padding: '0.625rem 0.75rem',
                        background: 'var(--bg-card-hover)',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-subtle)',
                        fontSize: '0.75rem',
                      }}
                    >
                      <div className="u-row" style={{ gap: '0.5rem', marginBottom: '0.25rem' }}>
                        <span
                          style={{
                            width: '18px',
                            height: '18px',
                            borderRadius: '50%',
                            background: 'rgba(99, 102, 241, 0.2)',
                            color: 'var(--primary)',
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.6875rem',
                            fontWeight: 700,
                          }}
                        >
                          {idx + 1}
                        </span>
                        <strong style={{ color: 'var(--text-primary)' }}>{st.name}</strong>
                      </div>
                      <p style={{ margin: 0, color: 'var(--text-tertiary)', fontSize: '0.6875rem' }}>
                        {st.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Existing project corpus (processed by the run even when nothing is queued) */}
              <div className="card" style={{ padding: '0.875rem 1rem' }} aria-label="Project corpus">
                <div className="u-row u-between u-wrap" style={{ gap: '0.5rem' }}>
                  <span className="form-label">Project Corpus ({corpusTotal} paper{corpusTotal === 1 ? '' : 's'})</span>
                  <span className="u-text-xs u-text-tertiary">Already stored; not downloaded again</span>
                </div>
                {corpus && corpusTotal > 0 ? (
                  <p className="u-text-sm u-text-secondary" style={{ marginTop: '0.375rem' }}>
                    {corpus.pending_extraction} with full text awaiting statement extraction ·{' '}
                    {corpus.statements_extracted} already extracted (reused) · {corpus.metadata_only} metadata only (no
                    statements possible). Gap discovery reads statements from all of this project's corpus papers.
                  </p>
                ) : (
                  <p className="u-text-sm u-text-secondary" style={{ marginTop: '0.375rem' }}>
                    No papers in this project's corpus yet.
                  </p>
                )}
              </div>

              {/* Selected Papers Table */}
              <div className="u-stack" style={{ gap: '0.5rem' }}>
                <div className="u-row u-between">
                  <span className="form-label">
                    Queued Papers ({data.papers.length})
                  </span>
                  <span className="u-text-xs u-text-tertiary">
                    {totalSelected === 0
                      ? 'No papers queued'
                      : 'Queued papers are ingested and tracked under the run record'}
                  </span>
                </div>

                {data.papers.length === 0 ? (
                  <div
                    style={{
                      padding: '2rem',
                      textAlign: 'center',
                      background: 'var(--bg-card-hover)',
                      borderRadius: 'var(--radius-md)',
                      border: '1px dashed var(--border-subtle)',
                    }}
                  >
                    <BookOpen size={24} style={{ color: 'var(--text-tertiary)', margin: '0 auto 0.5rem auto' }} />
                    <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                      {corpusTotal > 0
                        ? `No additional papers queued. The run will process the ${corpusTotal} corpus paper${corpusTotal === 1 ? '' : 's'} above.`
                        : 'No papers queued and the project corpus is empty.'}
                    </p>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                      To queue new papers for ingestion by the run, use “Select for Run” in Paper Search.
                    </p>
                  </div>
                ) : (
                  <div
                    style={{
                      maxHeight: '220px',
                      overflowY: 'auto',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-app)',
                    }}
                  >
                    <table
                      style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        fontSize: '0.75rem',
                        textAlign: 'left',
                      }}
                    >
                      <thead>
                        <tr
                          style={{
                            borderBottom: '1px solid var(--border-subtle)',
                            background: 'var(--bg-card-hover)',
                            color: 'var(--text-tertiary)',
                          }}
                        >
                          <th style={{ padding: '0.5rem 0.75rem' }}>Paper Title</th>
                          <th style={{ padding: '0.5rem 0.75rem' }}>Status</th>
                          <th style={{ padding: '0.5rem 0.75rem' }}>PDF Availability</th>
                          <th style={{ padding: '0.5rem 0.75rem' }}>Identifier</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.papers.map((p) => {
                          const pId = p.openalex_id || p.paper_id || '';
                          return (
                            <tr
                              key={pId}
                              style={{
                                borderBottom: '1px solid var(--border-subtle)',
                              }}
                            >
                              <td
                                style={{
                                  padding: '0.5rem 0.75rem',
                                  color: 'var(--text-primary)',
                                  maxWidth: '300px',
                                  whiteSpace: 'nowrap',
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                }}
                                title={p.title}
                              >
                                {p.title}
                              </td>
                              <td style={{ padding: '0.5rem 0.75rem', whiteSpace: 'nowrap' }}>
                                <Badge
                                  label={formatStatusLabel(p.status)}
                                  variant={getStatusBadgeVariant(p.status)}
                                />
                              </td>
                              <td style={{ padding: '0.5rem 0.75rem', whiteSpace: 'nowrap', color: p.has_pdf ? '#10b981' : '#f59e0b' }}>
                                {p.has_pdf ? 'Open-Access PDF' : 'No PDF Listed'}
                              </td>
                              <td
                                style={{
                                  padding: '0.5rem 0.75rem',
                                  fontFamily: 'monospace',
                                  color: 'var(--text-tertiary)',
                                  fontSize: '0.6875rem',
                                }}
                              >
                                {p.doi || pId}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <div className="modal-footer u-between">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onClose}
            disabled={submitting}
          >
            Cancel
          </button>

          <button
            type="button"
            className="btn btn-primary"
            onClick={handleConfirmRun}
            disabled={loading || submitting || !data || !runnable}
            title={data && !runnable ? 'Add papers to the project corpus or queue papers before starting a run.' : undefined}
            id="btn-confirm-start-run"
          >
            {submitting ? (
              <Loader2 size={16} className="spin" />
            ) : (
              <Play size={16} fill="currentColor" />
            )}
            <span>{submitting ? 'Initiating Run…' : 'Start Processing'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
