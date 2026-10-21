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
  ArrowRight,
  Sparkles
} from 'lucide-react';
import type { ConfigurationFingerprint, RunMetadata } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getScreeningDecisionBadgeVariant } from '../common/badgeVariants';
import { NotRecorded, RecordedValue } from '../common/NotRecorded';
import { displayValue } from '../../format';

interface RunOverviewViewProps {
  onNavigate: (route: string) => void;
}

const GENERIC_CAVEAT =
  'Results are model-generated and have not been validated by human experts; see run counts below.';

export const RunOverviewView: React.FC<RunOverviewViewProps> = ({ onNavigate }) => {
  const [metadata, setMetadata] = useState<RunMetadata | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetadata = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getRunMetadata()
      .then((data) => setMetadata(data))
      .catch((err: Error) => setError(err.message || 'Failed to fetch experiment run metadata.'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchMetadata();
  }, [fetchMetadata]);

  const rankedPapers = useMemo(() => {
    const ranked = metadata?.relevance_screening?.ranked_papers;
    if (!Array.isArray(ranked)) return [];
    return [...ranked].sort((a, b) => a.rank - b.rank);
  }, [metadata]);

  if (loading) {
    return <LoadingState message="Loading completed experiment configuration & metrics..." minHeight="400px" />;
  }

  if (error || !metadata) {
    return (
      <div className="u-stack-lg">
        <ErrorBanner
          title="Could Not Load Run Metadata"
          message={error || 'No metadata returned from experiment database.'}
          onRetry={fetchMetadata}
        />
      </div>
    );
  }

  const fp: ConfigurationFingerprint | null = metadata.run_fingerprint;
  const screening = metadata.relevance_screening;
  const counts = metadata.counts;
  const screeningConfigured = fp?.relevance_screening?.enabled;

  return (
    <div className="view-container animate-fade-in">
      {/* View Header */}
      <div className="view-header">
        <div>
          <h1 className="view-title">Completed Experiment Run Overview</h1>
          <p className="view-subtitle">
            Configuration parameters, temporal cutoffs, relevance screening decisions, and corpus entities.
          </p>
        </div>
        <button
          onClick={fetchMetadata}
          className="btn btn-secondary"
          title="Refresh metadata from read-only database"
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Methodological Limitations Callout */}
      <div className="callout callout-warning">
        <div className="callout-icon-col">
          <AlertTriangle size={20} />
        </div>
        <div className="callout-content">
          <h4 className="callout-title">Experimental Scope & Methodological Limitations</h4>
          {metadata.metric_interpretation_note && (
            <p className="callout-text">{metadata.metric_interpretation_note}</p>
          )}
          <p className={metadata.metric_interpretation_note ? 'callout-subtext' : 'callout-text'}>
            {GENERIC_CAVEAT}
          </p>
        </div>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid-4-cols">
        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-label">Corpus Papers</span>
            <FileText className="stat-icon tone-indigo" />
          </div>
          <div className="stat-value">{counts ? counts.papers : metadata.total_papers}</div>
          <div className="stat-sublabel">
            {counts ? (
              <>
                <span className="tone-emerald u-text-strong">{counts.baseline_papers}</span> baseline ·{' '}
                <span className="tone-cyan u-text-strong">{counts.later_evaluation_papers}</span> later-evaluation
              </>
            ) : (
              <NotRecorded label="Baseline / later split not recorded" />
            )}
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-label">Extracted Statements</span>
            <Layers className="stat-icon tone-cyan" />
          </div>
          <div className="stat-value">{counts ? counts.statements : metadata.total_statements}</div>
          <div className="stat-sublabel">Statement records in database</div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-label">Research Gaps</span>
            <Sparkles className="stat-icon tone-amber" />
          </div>
          <div className="stat-value">{counts ? counts.research_gaps : metadata.total_gaps}</div>
          <div className="stat-sublabel">Research gap records in database</div>
        </div>

        <div className="stat-card">
          <div className="stat-card-header">
            <span className="stat-label">Opportunities</span>
            <Activity className="stat-icon tone-indigo" />
          </div>
          <div className="stat-value">{counts ? counts.opportunities : metadata.total_opportunities}</div>
          <div className="stat-sublabel">
            {counts ? (
              <>
                <span className="tone-indigo u-text-strong">{counts.evidence_records}</span> temporal evidence records
              </>
            ) : (
              <NotRecorded label="Evidence record count not recorded" />
            )}
          </div>
        </div>
      </div>

      {/* Two Column Configuration Breakdown */}
      <div className="grid-2-cols">
        {/* Left: Execution & Model Fingerprint */}
        <div className="card">
          <div className="card-header">
            <div className="u-row">
              <Cpu size={16} className="tone-indigo" />
              <h3 className="card-title">Run Configuration & Fingerprint</h3>
            </div>
            <span className="card-header-meta">
              {fp ? 'Stored configuration fingerprint' : 'Fingerprint not recorded'}
            </span>
          </div>
          <div className="card-body">
            <div className="info-table">
              <div className="info-row">
                <span className="info-key">LLM Provider</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.llm_provider} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">LLM Model</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.llm_model} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Embedding Provider</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.embedding_provider} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Embedding Model</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.embedding_model} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Embedding Dimension</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.embedding_dimension} suffix="dims" /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Extraction Schema Version</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.extraction_schema_version} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Passage Selection (max passages / max words)</span>
                <span className="info-value u-mono">
                  <RecordedValue value={fp?.passage_selection?.max_passages} />
                  {' / '}
                  <RecordedValue value={fp?.passage_selection?.max_total_words} />
                </span>
              </div>
              <div className="info-row">
                <span className="info-key">Stage 3 Similarity Threshold</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.gap_discovery?.similarity_threshold} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Min Supporting Evidence</span>
                <span className="info-value u-mono">
                  <RecordedValue value={fp?.gap_discovery?.min_supporting_evidence} suffix="statements" />
                </span>
              </div>
              <div className="info-row">
                <span className="info-key">Max Refinement Iterations</span>
                <span className="info-value u-mono"><RecordedValue value={fp?.max_refinement_iterations} /></span>
              </div>
              <div className="info-row">
                <span className="info-key">Database Status (as reported by API)</span>
                <span className="info-value">
                  <span className="status-pill status-pill-neutral">
                    <RecordedValue value={metadata.database_status} />
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Temporal Cutoffs & Evolution Windows */}
        <div className="card">
          <div className="card-header">
            <div className="u-row">
              <Calendar size={16} className="tone-cyan" />
              <h3 className="card-title">Temporal Cutoff & Evolution Schedule</h3>
            </div>
            {metadata.baseline_cutoff ? (
              <span className="app-badge badge-temporal-baseline">Cutoff: {metadata.baseline_cutoff}</span>
            ) : (
              <span className="app-badge badge-neutral">Cutoff: Not recorded</span>
            )}
          </div>
          <div className="card-body">
            <div className="info-table">
              <div className="info-row">
                <span className="info-key">Baseline Discovery Cutoff</span>
                <span className="info-value u-mono tone-emerald">
                  <RecordedValue value={metadata.baseline_cutoff} />
                </span>
              </div>
              <div className="info-row">
                <span className="info-key">Stage 5 Evolution Enabled</span>
                <span className="info-value">
                  {!fp ? (
                    <NotRecorded />
                  ) : metadata.stage_5_enabled ? (
                    <span className="status-pill status-pill-success">Enabled</span>
                  ) : (
                    <span className="status-pill status-pill-neutral">Disabled</span>
                  )}
                </span>
              </div>
              <div className="info-row">
                <span className="info-key">Batch Window Size</span>
                <span className="info-value u-mono">
                  <RecordedValue value={metadata.stage_5_batch_window_days} suffix="days" />
                </span>
              </div>
            </div>

            {/* Evaluation Batches Timeline */}
            <div className="section-divider">
              <span className="label-caps">Stage 5 Evaluation Batch Windows</span>
              {metadata.evaluation_cutoffs && metadata.evaluation_cutoffs.length > 0 ? (
                <div className="timeline-mini">
                  {metadata.evaluation_cutoffs.map((cutoff, idx) => (
                    <div key={`${cutoff}-${idx}`} className="timeline-mini-item">
                      <div className="timeline-mini-bullet" />
                      <div className="timeline-mini-content">
                        <span className="timeline-mini-batch">Batch {idx + 1}</span>
                        <span className="timeline-mini-date">{cutoff}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="empty-note">No evaluation batch cutoffs recorded.</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Relevance Screening Summary */}
      <div className="card">
        <div className="card-header">
          <div className="u-row">
            <Filter size={16} className="tone-purple" />
            <h3 className="card-title">Pre-Extraction Relevance Screening</h3>
          </div>
          {screening && (
            <span className="card-header-meta">
              Topic: {displayValue(screening.topic) ? `"${screening.topic}"` : <NotRecorded />}
            </span>
          )}
        </div>
        <div className="card-body">
          {!screening ? (
            <p className="empty-note">
              {screeningConfigured === false
                ? 'Relevance screening was not enabled in the recorded run configuration.'
                : 'No relevance screening report was recorded for this run.'}
            </p>
          ) : (
            <>
              <div className="grid-3-cols">
                <div className="sub-stat-box">
                  <span className="sub-stat-label">Similarity Threshold</span>
                  <span className="sub-stat-val u-mono"><RecordedValue value={screening.threshold} /></span>
                  <span className="sub-stat-note">
                    Min papers: <RecordedValue value={screening.min_papers} /> · Max papers:{' '}
                    <RecordedValue value={screening.max_papers} missingLabel="No limit recorded" />
                  </span>
                </div>
                <div className="sub-stat-box">
                  <span className="sub-stat-label">Eligible Papers</span>
                  <span className="sub-stat-val u-mono"><RecordedValue value={screening.total_eligible_papers} /></span>
                </div>
                <div className="sub-stat-box">
                  <span className="sub-stat-label">Screening Decisions</span>
                  <span className="sub-stat-val u-mono">
                    <span className="tone-emerald"><RecordedValue value={screening.selected_count} /> Selected</span>
                    {' · '}
                    <span className="tone-rose"><RecordedValue value={screening.excluded_count} /> Excluded</span>
                  </span>
                </div>
              </div>

              {screening.floor_warning && (
                <p className="screening-floor-warning">{screening.floor_warning}</p>
              )}
              {screening.metric_interpretation_note && (
                <p className="callout-subtext">{screening.metric_interpretation_note}</p>
              )}

              {rankedPapers.length > 0 ? (
                <div className="screening-decisions-table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Paper / Title</th>
                        <th>Score</th>
                        <th>Decision</th>
                        <th>Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rankedPapers.map((rp) => (
                        <tr key={`${rp.paper_id}-${rp.rank}`}>
                          <td className="u-mono u-text-sm">{rp.rank}</td>
                          <td className="u-text-strong">
                            {rp.title || rp.filename || <NotRecorded />}
                            {rp.title && rp.filename && (
                              <span className="table-subline">{rp.filename}</span>
                            )}
                            {rp.publication_date && (
                              <span className="table-subline">{rp.publication_date}</span>
                            )}
                          </td>
                          <td className="u-mono u-text-sm">
                            {typeof rp.score === 'number' ? rp.score.toFixed(4) : <NotRecorded />}
                          </td>
                          <td>
                            {rp.decision ? (
                              <Badge
                                label={rp.decision}
                                variant={getScreeningDecisionBadgeVariant(rp.decision)}
                              />
                            ) : (
                              <NotRecorded />
                            )}
                          </td>
                          <td className="u-text-sm u-text-secondary">
                            <RecordedValue value={rp.reason} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="empty-note">The screening report contains no ranked paper entries.</p>
              )}
            </>
          )}
        </div>
      </div>

      {/* Next View Call to Action */}
      <div className="nav-callout-card">
        <div>
          <h4 className="nav-callout-title">Explore Research Opportunities</h4>
          <p className="nav-callout-desc">
            Browse the {counts ? counts.opportunities : metadata.total_opportunities} generated opportunity proposals and inspect their chronological evolution evidence.
          </p>
        </div>
        <button
          onClick={() => onNavigate('#/opportunities')}
          className="btn btn-primary"
        >
          <span>Open Opportunities Explorer</span>
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
};
