import React, { useEffect, useState, useRef } from 'react';
import {
  X,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Loader2,
  Terminal,
  FileText,
  Lightbulb,
  Compass,
  ArrowRight,
  BookOpen,
  Copy,
  Check,
  AlertTriangle,
} from 'lucide-react';
import { api } from '../../api/client';
import type { ResearchRun, RunStage, RunPaper } from '../../types';
import { Badge } from '../common/Badge';
import type { BadgeVariant } from '../common/badgeVariants';
import { reportRunStatus, registerMonitoredRun } from '../../workflow/runMonitor';

interface RunProgressModalProps {
  isOpen: boolean;
  onClose: () => void;
  runId: string | null;
  projectId: string;
  onNavigate: (route: string) => void;
}

export const RunProgressModal: React.FC<RunProgressModalProps> = ({
  isOpen,
  onClose,
  runId,
  projectId,
  onNavigate,
}) => {
  const [run, setRun] = useState<ResearchRun | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState<boolean>(false);
  const [pollEpoch, setPollEpoch] = useState<number>(0);

  // Technical logs view state
  const [showLogs, setShowLogs] = useState<boolean>(false);
  const [logsText, setLogsText] = useState<string>('');
  const [loadingLogs, setLoadingLogs] = useState<boolean>(false);
  const [copiedLogs, setCopiedLogs] = useState<boolean>(false);

  const pollTimer = useRef<number | null>(null);

  const fetchStatus = async (isInitial = false) => {
    if (!runId) return;
    if (isInitial) setLoading(true);
    try {
      const data = await api.getRunStatus(runId);
      setRun(data);
      reportRunStatus(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to poll run status.');
    } finally {
      if (isInitial) setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen || !runId) {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
      return;
    }

    // Full loading state only on open; a retry restart refreshes silently.
    fetchStatus(pollEpoch === 0);

    const schedulePoll = () => {
      pollTimer.current = window.setTimeout(async () => {
        if (!isOpen || !runId) return;
        try {
          const data = await api.getRunStatus(runId);
          setRun(data);
          reportRunStatus(data);
          // If still running or pending, continue polling every 2 seconds
          if (data.status === 'RUNNING' || data.status === 'PENDING') {
            schedulePoll();
          }
        } catch {
          schedulePoll();
        }
      }, 2000);
    };

    schedulePoll();

    return () => {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, [isOpen, runId, pollEpoch]);

  const handleRetry = async () => {
    if (!runId || retrying) return;
    setRetrying(true);
    try {
      const updatedRun = await api.retryRun(runId);
      setRun(updatedRun);
      registerMonitoredRun(runId, projectId);
      reportRunStatus(updatedRun);
      // Polling stopped when the run reached a terminal state; restart it for the retried run.
      setPollEpoch((n) => n + 1);
    } catch (err: any) {
      setError(err.message || 'Failed to trigger run retry.');
    } finally {
      setRetrying(false);
    }
  };

  const handleOpenLogs = async () => {
    if (!runId) return;
    setShowLogs(true);
    setLoadingLogs(true);
    try {
      const logsResp = await api.getRunLogs(runId);
      const formatted = (logsResp.logs || []).map((l: any) =>
        typeof l === 'string'
          ? l
          : `[${l.timestamp || ''}] [${l.level || 'INFO'}] [${l.stage || 'GENERAL'}] ${l.message || ''}`,
      );
      setLogsText(formatted.join('\n'));
    } catch (err: any) {
      setLogsText(`Error fetching logs: ${err.message || 'Unknown error'}`);
    } finally {
      setLoadingLogs(false);
    }
  };

  const handleCopyLogs = () => {
    if (!logsText) return;
    navigator.clipboard.writeText(logsText);
    setCopiedLogs(true);
    setTimeout(() => setCopiedLogs(false), 2000);
  };

  if (!isOpen) return null;

  const isTerminal =
    run &&
    ['COMPLETED', 'PARTIAL_SUCCESS', 'FAILED', 'AWAITING_REVIEW', 'INTERRUPTED'].includes(run.status);

  const getStageIcon = (stageName: string) => {
    switch (stageName) {
      case 'INGESTION':
        return <BookOpen size={16} />;
      case 'UNDERSTANDING':
        return <FileText size={16} />;
      case 'GAP_DISCOVERY':
        return <Lightbulb size={16} />;
      case 'OPPORTUNITY_GENERATION':
        return <Compass size={16} />;
      default:
        return <Play size={16} />;
    }
  };

  const getRunStatusBadgeVariant = (status: string): BadgeVariant => {
    switch (status) {
      case 'COMPLETED':
        return 'status-positive';
      case 'AWAITING_REVIEW':
      case 'PARTIAL_SUCCESS':
        return 'status-warning';
      case 'FAILED':
      case 'INTERRUPTED':
        return 'status-negative';
      default:
        return 'neutral';
    }
  };

  const getPaperStatusBadgeVariant = (status: string): BadgeVariant => {
    switch (status) {
      case 'INGESTED':
      case 'COMPLETED':
      case 'SKIPPED_DUPLICATE':
        return 'status-positive';
      case 'NO_PDF':
      case 'METADATA_ONLY':
        return 'status-warning';
      case 'FAILED':
      case 'ERROR':
      case 'INTERRUPTED':
        return 'status-negative';
      default:
        return 'neutral';
    }
  };

  const stagesList: RunStage[] =
    run?.stages && run.stages.length > 0
      ? run.stages
      : run?.stage_progress?.stages || [];

  const summaryData = run?.summary;
  const successfulPapers = summaryData?.successful_papers ?? 0;
  const failedPapers = summaryData?.failed_papers ?? 0;
  const nextActions: string[] = summaryData?.next_actions || [];

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="run-progress-title"
    >
      <div
        className="modal-dialog modal-dialog-lg"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '860px', maxHeight: '92vh' }}
      >
        {/* Header */}
        <div className="modal-header">
          <div className="u-row" style={{ gap: '0.875rem' }}>
            <div className="modal-icon-badge modal-icon-badge-primary">
              <Play size={20} />
            </div>
            <div>
              <div className="u-row" style={{ gap: '0.5rem', alignItems: 'center' }}>
                <h2 id="run-progress-title" className="modal-title">
                  Research Pipeline Execution
                </h2>
                {run && (
                  <Badge
                    label={run.status.replace('_', ' ')}
                    variant={getRunStatusBadgeVariant(run.status)}
                  />
                )}
              </div>
              <p className="modal-subtitle">
                Run ID: <span className="u-mono">{runId}</span>
              </p>
            </div>
          </div>

          <div className="u-row" style={{ gap: '0.5rem' }}>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleOpenLogs}
              title="Inspect low-level execution logs"
              id="btn-view-run-logs"
            >
              <Terminal size={14} />
              <span>Technical Logs</span>
            </button>
            <button
              type="button"
              className="btn-icon"
              onClick={onClose}
              aria-label="Close run monitor"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="modal-body u-stack" style={{ gap: '1.25rem', overflowY: 'auto' }}>
          {loading && !run && (
            <div className="inline-pending" style={{ padding: '3rem 1rem', justifyContent: 'center' }}>
              <Loader2 size={24} className="spin" />
              <span>Loading execution status and per-paper checkpoints…</span>
            </div>
          )}

          {error && (
            <div className="action-result is-error" role="alert">
              <p className="action-result-title">Status Error</p>
              <p className="u-text-sm">{error}</p>
            </div>
          )}

          {run && (
            <>
              {/* Sequential Stage Progress Flow */}
              <div className="u-stack" style={{ gap: '0.625rem' }}>
                <span className="form-label">Pipeline Stage Progression</span>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                    gap: '0.625rem',
                  }}
                >
                  {stagesList.map((st, idx) => {
                    const isDone = st.status === 'COMPLETED';
                    const isRunning = st.status === 'RUNNING';
                    const isFailed = st.status === 'FAILED';
                    const stageKey = st.stage_id || st.stage || `stage_${idx}`;

                    return (
                      <div
                        key={stageKey}
                        style={{
                          padding: '0.75rem',
                          borderRadius: 'var(--radius-md)',
                          background: isRunning
                            ? 'rgba(99, 102, 241, 0.08)'
                            : isDone
                              ? 'rgba(16, 185, 129, 0.05)'
                              : isFailed
                                ? 'rgba(239, 68, 68, 0.08)'
                                : 'var(--bg-card-hover)',
                          border: `1px solid ${
                            isRunning
                              ? 'var(--primary)'
                              : isDone
                                ? 'rgba(16, 185, 129, 0.3)'
                                : isFailed
                                  ? 'rgba(239, 68, 68, 0.4)'
                                  : 'var(--border-subtle)'
                          }`,
                          display: 'flex',
                          flexDirection: 'column',
                          gap: '0.375rem',
                        }}
                      >
                        <div className="u-row u-between" style={{ alignItems: 'center' }}>
                          <span
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.375rem',
                              fontWeight: 600,
                              fontSize: '0.8125rem',
                              color: isRunning
                                ? 'var(--primary)'
                                : isDone
                                  ? '#10b981'
                                  : isFailed
                                    ? '#ef4444'
                                    : 'var(--text-secondary)',
                            }}
                          >
                            {getStageIcon(st.stage_id || st.stage || '')}
                            <span>{st.name}</span>
                          </span>
                          {isRunning && <Loader2 size={13} className="spin" style={{ color: 'var(--primary)' }} />}
                          {isDone && <CheckCircle2 size={14} style={{ color: '#10b981' }} />}
                          {isFailed && <AlertCircle size={14} style={{ color: '#ef4444' }} />}
                          {!isRunning && !isDone && !isFailed && (
                            <Clock size={13} style={{ color: 'var(--text-tertiary)' }} />
                          )}
                        </div>

                        <span
                          style={{
                            fontSize: '0.6875rem',
                            color: 'var(--text-tertiary)',
                            textTransform: 'uppercase',
                            letterSpacing: '0.05em',
                          }}
                        >
                          {st.status}
                        </span>

                        {(st.error || st.message) && (
                          <span style={{ fontSize: '0.6875rem', color: isFailed ? '#ef4444' : 'var(--text-secondary)', marginTop: '0.25rem', overflowWrap: 'anywhere' }}>
                            {st.error || st.message}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Per-Paper Status Breakdown */}
              <div className="u-stack" style={{ gap: '0.5rem' }}>
                <div className="u-row u-between">
                  <span className="form-label">
                    Paper-Level Checkpoints ({run.papers.length})
                  </span>
                  <span className="u-text-xs u-text-tertiary">
                    Fault tolerant: errors on individual papers do not halt the run
                  </span>
                </div>

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
                        <th style={{ padding: '0.5rem 0.75rem' }}>Passages</th>
                        <th style={{ padding: '0.5rem 0.75rem' }}>Statements</th>
                        <th style={{ padding: '0.5rem 0.75rem' }}>Diagnostic Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {run.papers.map((p: RunPaper) => {
                        const paperKey = p.openalex_id || p.paper_id || p.id;
                        return (
                          <tr
                            key={paperKey}
                            style={{
                              borderBottom: '1px solid var(--border-subtle)',
                            }}
                          >
                            <td
                              style={{
                                padding: '0.5rem 0.75rem',
                                color: 'var(--text-primary)',
                                maxWidth: '260px',
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
                                label={p.status}
                                variant={getPaperStatusBadgeVariant(p.status)}
                              />
                            </td>
                            <td style={{ padding: '0.5rem 0.75rem', whiteSpace: 'nowrap', color: 'var(--text-secondary)' }}>
                              {p.passages_count ?? 0}
                            </td>
                            <td style={{ padding: '0.5rem 0.75rem', whiteSpace: 'nowrap', color: 'var(--text-secondary)' }}>
                              {p.statements_count ?? 0}
                            </td>
                            <td
                              style={{
                                padding: '0.5rem 0.75rem',
                                fontSize: '0.6875rem',
                                color: p.error_message ? '#ef4444' : 'var(--text-tertiary)',
                                maxWidth: '220px',
                                whiteSpace: 'nowrap',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                              }}
                              title={p.error_message || ''}
                            >
                              {p.error_message || (p.status === 'COMPLETED' ? `${p.statements_count} statement(s)` : '—')}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Completion & Next Actions Summary */}
              {isTerminal && (
                <div
                  style={{
                    padding: '1rem',
                    borderRadius: 'var(--radius-md)',
                    background:
                      run.status === 'COMPLETED' || run.status === 'AWAITING_REVIEW'
                        ? 'rgba(16, 185, 129, 0.06)'
                        : 'rgba(239, 68, 68, 0.06)',
                    border: `1px solid ${
                      run.status === 'COMPLETED' || run.status === 'AWAITING_REVIEW'
                        ? 'rgba(16, 185, 129, 0.3)'
                        : 'rgba(239, 68, 68, 0.3)'
                    }`,
                  }}
                >
                  <div className="u-row u-between" style={{ alignItems: 'flex-start' }}>
                    <div>
                      <div className="u-row" style={{ gap: '0.5rem', alignItems: 'center' }}>
                        {run.status === 'COMPLETED' || run.status === 'AWAITING_REVIEW' ? (
                          <CheckCircle2 size={16} style={{ color: '#10b981' }} />
                        ) : (
                          <AlertTriangle size={16} style={{ color: '#ef4444' }} />
                        )}
                        <strong style={{ fontSize: '0.875rem', color: 'var(--text-primary)' }}>
                          {run.status === 'AWAITING_REVIEW'
                            ? 'Run Complete — Awaiting Researcher Review'
                            : run.status === 'COMPLETED'
                              ? 'Research Pipeline Complete'
                              : 'Run Finished with Diagnostic Issues'}
                        </strong>
                      </div>
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        Processed {successfulPapers} successful paper(s) (
                        {failedPapers} failed or skipped). Work is safely persisted.
                      </p>
                      {run.error_message && (
                        <p className="inline-note is-error" role="alert">{run.error_message}</p>
                      )}
                    </div>

                    {(run.status === 'PARTIAL_SUCCESS' || run.status === 'FAILED' || run.status === 'INTERRUPTED' ||
                      (run.status === 'AWAITING_REVIEW' && failedPapers > 0)) && (
                      <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        onClick={handleRetry}
                        disabled={retrying}
                        title="Safely retry failed papers without duplicating data"
                        id="btn-retry-run"
                      >
                        {retrying ? <Loader2 size={13} className="spin" /> : <RotateCcw size={13} />}
                        <span>{retrying ? 'Retrying…' : 'Retry Failed Stages'}</span>
                      </button>
                    )}
                  </div>

                  {nextActions && nextActions.length > 0 && (
                    <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)' }}>
                      <span className="u-text-xs" style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>
                        Recommended Next Actions:
                      </span>
                      <ul className="plain-list u-text-xs" style={{ marginTop: '0.375rem', color: 'var(--text-secondary)' }}>
                        {nextActions.map((a) => (
                          <li key={a}>{a}</li>
                        ))}
                      </ul>
                      <div className="u-row u-wrap" style={{ gap: '0.5rem', marginTop: '0.5rem' }}>
                        <button
                          type="button"
                          className={`btn ${run.status === 'AWAITING_REVIEW' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                          onClick={() => {
                            onClose();
                            onNavigate(`#/projects/${projectId}/gaps`);
                          }}
                        >
                          <Lightbulb size={13} />
                          <span>Review Gaps</span>
                          <ArrowRight size={12} />
                        </button>

                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => {
                            onClose();
                            onNavigate(`#/projects/${projectId}/opportunities`);
                          }}
                        >
                          <Compass size={13} />
                          <span>Opportunities</span>
                          <ArrowRight size={12} />
                        </button>

                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => {
                            onClose();
                            onNavigate(`#/projects/${projectId}/papers?tab=corpus`);
                          }}
                        >
                          <BookOpen size={13} />
                          <span>Corpus Inspector</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer u-between">
          <span className="u-text-xs u-text-tertiary">
            {!isTerminal ? 'Polling execution progress every 2 seconds…' : 'Execution completed.'}
          </span>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onClose}>
            {isTerminal ? 'Close' : 'Minimize (Run continues in background)'}
          </button>
        </div>

        {/* Technical Logs Drawer / Overlay */}
        {showLogs && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              background: 'var(--bg-card)',
              zIndex: 10,
              borderRadius: 'var(--radius-xl)',
              display: 'flex',
              flexDirection: 'column',
              boxShadow: 'var(--shadow-xl)',
            }}
          >
            <div className="modal-header">
              <div className="u-row" style={{ gap: '0.5rem', alignItems: 'center' }}>
                <Terminal size={18} style={{ color: 'var(--primary)' }} />
                <h3 className="modal-title" style={{ fontSize: '1rem' }}>
                  Technical Execution Logs
                </h3>
              </div>
              <div className="u-row" style={{ gap: '0.5rem' }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleCopyLogs}
                  disabled={!logsText}
                >
                  {copiedLogs ? <Check size={13} /> : <Copy size={13} />}
                  <span>{copiedLogs ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  type="button"
                  className="btn-icon"
                  onClick={() => setShowLogs(false)}
                  aria-label="Close logs drawer"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            <div style={{ flex: 1, padding: '1rem', background: '#0a0d14', overflowY: 'auto' }}>
              {loadingLogs ? (
                <div className="inline-pending" style={{ color: '#94a3b8' }}>
                  <Loader2 size={16} className="spin" />
                  <span>Fetching execution logs…</span>
                </div>
              ) : (
                <pre
                  style={{
                    margin: 0,
                    fontFamily: 'monospace',
                    fontSize: '0.75rem',
                    lineHeight: '1.5',
                    color: '#e2e8f0',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                  }}
                >
                  {logsText || 'No logs available.'}
                </pre>
              )}
            </div>

            <div className="modal-footer u-between">
              <span className="u-text-xs u-text-tertiary">
                Direct execution stream from backend runner
              </span>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setShowLogs(false)}
              >
                Back to Progress Overview
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
