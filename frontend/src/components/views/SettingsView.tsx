import { useState, useCallback, useEffect } from 'react';
import { Cpu, CheckCircle2, XCircle, Terminal, RefreshCw, ShieldCheck, Sparkles } from 'lucide-react';
import { api } from '../../api/client';
import type { SystemStatus } from '../../types';
import { ErrorBanner } from '../common/ErrorBanner';
import { describeModelStatus, describeEndpointStatus } from '../common/providerStatus';

function StatusBadge({ ok, okLabel, badLabel }: { ok: boolean | null; okLabel: string; badLabel: string }) {
  if (ok === null) return <span className="badge badge-default">Checking...</span>;
  return ok ? (
    <span className="badge badge-success">
      <CheckCircle2 size={10} aria-hidden="true" /> {okLabel}
    </span>
  ) : (
    <span className="badge badge-danger">
      <XCircle size={10} aria-hidden="true" /> {badLabel}
    </span>
  );
}

function DescriptorBadge({ label, variant, tooltip }: { label: string; variant: 'success' | 'danger' | 'default' | 'info' | 'warning'; tooltip?: string }) {
  return (
    <span className={`badge badge-${variant}`} title={tooltip}>
      {variant === 'success' && <CheckCircle2 size={10} aria-hidden="true" />}
      {variant === 'danger' && <XCircle size={10} aria-hidden="true" />}
      {label}
    </span>
  );
}

export const SettingsView: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState<boolean>(false);

  const refresh = useCallback(async () => {
    setChecking(true);
    try {
      setStatus(await api.getSystemStatus());
      setError(null);
    } catch (err: unknown) {
      setStatus(null);
      setError(err instanceof Error ? err.message : 'Could not load system status.');
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const ollama = status?.ollama ?? null;

  return (
    <div className="view-container animate-fade-in">
      <div className="view-header">
        <div>
          <div className="u-row" style={{ gap: '0.625rem', marginBottom: '0.25rem' }}>
            <span className="badge badge-default">System Configuration</span>
            <span className="badge badge-primary">Local Research Environment</span>
          </div>
          <h1 className="view-title">Application Settings &amp; Diagnostics</h1>
          <p className="view-subtitle">
            Configuration reported by the running backend, with live availability checks. Values are read-only here;
            change them through the backend environment (<code className="font-mono">IDEAGEN_*</code> variables).
          </p>
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => void refresh()} disabled={checking}>
          <RefreshCw size={14} className={checking ? 'spin' : ''} aria-hidden="true" />
          <span>{checking ? 'Checking...' : 'Refresh Status'}</span>
        </button>
      </div>

      {error && <ErrorBanner title="Backend status unavailable" message={error} onRetry={() => void refresh()} />}

      <div className="grid-2-cols">
        <div className="card u-stack-md">
          <div className="u-row" style={{ gap: '0.75rem' }}>
            <div className="modal-icon-badge modal-icon-badge-primary">
              <Terminal size={20} aria-hidden="true" />
            </div>
            <div>
              <h2 className="card-title">Runtime Infrastructure</h2>
              <span className="text-secondary">Backend, database and literature source</span>
            </div>
          </div>

          <div className="u-stack-sm">
            <div className="detail-row">
              <span className="detail-label">IdeaGen API</span>
              <span className="detail-value u-row" style={{ gap: '0.5rem' }}>
                <span className="font-mono">/api (dev proxy)</span>
                <StatusBadge ok={error ? false : status ? true : null} okLabel="Connected" badLabel="Unreachable" />
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Database</span>
              <span className="detail-value u-row" style={{ gap: '0.5rem' }}>
                <span className="font-mono">{status?.database.dialect ?? '—'}</span>
                <StatusBadge ok={status ? status.database.connected : null} okLabel="Connected" badLabel="Error" />
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">OpenAlex API</span>
              <span className="detail-value">
                {status
                  ? status.openalex.proxy
                    ? `Requests routed through proxy ${status.openalex.proxy}`
                    : 'Direct connection (not checked until you search)'
                  : '—'}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">OpenAlex polite-pool email</span>
              <span className="detail-value">{status ? (status.openalex.email_configured ? 'Configured' : 'Not configured') : '—'}</span>
            </div>
          </div>
        </div>

        <div className="card u-stack-md">
          <div className="u-row" style={{ gap: '0.75rem' }}>
            <div className="modal-icon-badge">
              <Cpu size={20} aria-hidden="true" />
            </div>
            <div>
              <h2 className="card-title">LLM &amp; Embeddings</h2>
              <span className="text-secondary">Configured providers and model availability</span>
            </div>
          </div>

          <div className="u-stack-sm">
            <div className="detail-row">
              <span className="detail-label">LLM ({status?.llm.provider ?? '—'})</span>
              <span className="detail-value u-row" style={{ gap: '0.5rem' }}>
                <span className="font-mono">{status?.llm.model ?? '—'}</span>
                {(() => {
                  const desc = describeModelStatus(status, status?.llm.model ?? null, status?.llm.provider ?? 'unknown');
                  return <DescriptorBadge label={desc.badgeLabel} variant={desc.badgeVariant} tooltip={desc.tooltip} />;
                })()}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Embeddings ({status?.embedding.provider ?? '—'})</span>
              <span className="detail-value u-row" style={{ gap: '0.5rem' }}>
                <span className="font-mono">
                  {status?.embedding.model ?? '—'}
                  {status?.embedding.dimension ? ` (${status.embedding.dimension}-dim)` : ''}
                </span>
                {(() => {
                  const desc = describeModelStatus(status, status?.embedding.model ?? null, status?.embedding.provider ?? 'unknown');
                  return <DescriptorBadge label={desc.badgeLabel} variant={desc.badgeVariant} tooltip={desc.tooltip} />;
                })()}
              </span>
            </div>
            {ollama && (() => {
              const ep = describeEndpointStatus(status);
              return (
                <div className="detail-row">
                  <span className="detail-label">Ollama endpoint</span>
                  <span className="detail-value u-row" style={{ gap: '0.5rem' }}>
                    <span className="font-mono">{status?.llm.base_url ?? status?.embedding.base_url}</span>
                    <DescriptorBadge label={ep.badgeLabel} variant={ep.badgeVariant} tooltip={ep.tooltip} />
                  </span>
                </div>
              );
            })()}
            {ollama && !ollama.reachable && ollama.auto_manage && !ollama.process_running ? (
              <div className="callout callout-info" style={{ marginTop: '0.5rem', padding: '0.75rem 1rem' }}>
                <div className="u-row" style={{ gap: '0.5rem', fontSize: '0.8125rem' }}>
                  <Sparkles size={14} className="text-primary" style={{ flexShrink: 0 }} />
                  <span>
                    Ollama lifecycle is <strong>auto-managed</strong>. The daemon is currently idle and starts automatically when a research run begins (idle timeout: {ollama.idle_timeout ?? 60}s).
                  </span>
                </div>
              </div>
            ) : ollama?.error && (!ollama.auto_manage || ollama.process_running) ? (
              <p className="inline-note is-error" role="alert">{ollama.error}</p>
            ) : null}
            {ollama && ollama.reachable && ollama.missing_models.length > 0 && (
              <p className="inline-note is-error" role="alert">
                Missing models: {ollama.missing_models.join(', ')}. Install with{' '}
                <code className="font-mono">ollama pull {ollama.missing_models[0]}</code>. Statement extraction, gap
                discovery and opportunity generation fail until the configured models are available.
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="callout callout-info">
        <div className="callout-icon-col">
          <ShieldCheck size={20} aria-hidden="true" />
        </div>
        <div className="callout-content">
          <div className="callout-title">Architecture &amp; Data Scoping</div>
          <p className="callout-body">
            <strong>Authentication:</strong> none. IdeaGen is a single-user local workspace.
            <br />
            <strong>Project scoping:</strong> projects are stored by the backend. Papers are shared in one literature
            store and linked to projects (<code className="font-mono">project_papers</code>); gaps, opportunities and runs
            carry a <code className="font-mono">project_id</code>. Gap discovery and opportunity prior-art retrieval only use
            the active project's papers.
            <br />
            <strong>Model outputs:</strong> generated statements, gaps and opportunities are only as reliable as the
            configured local models; they are grounded in stored passages, not independently verified facts.
          </p>
        </div>
      </div>
    </div>
  );
};
