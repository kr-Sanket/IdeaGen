import React, { useEffect, useState, useCallback } from 'react';
import { AlertTriangle, RefreshCw, Compass, ArrowRight } from 'lucide-react';
import type { OpportunitySummary, OpportunityEvidenceItem, OpportunityHistoryItem, ResearchProject } from '../../types';
import { api } from '../../api/client';
import { LoadingState } from '../common/LoadingState';
import { ErrorBanner } from '../common/ErrorBanner';
import { Badge } from '../common/Badge';
import { getLifecycleBadgeVariant, getRelationshipBadgeVariant } from '../common/badgeVariants';
import { EvolvePanel } from '../workflow/EvolvePanel';

interface EvolutionViewProps {
  project: ResearchProject;
  onNavigate: (route: string) => void;
}

export const EvolutionView: React.FC<EvolutionViewProps> = ({ project, onNavigate }) => {
  const [opportunities, setOpportunities] = useState<OpportunitySummary[]>([]);
  const [selectedOppId, setSelectedOppId] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<OpportunityEvidenceItem[]>([]);
  const [history, setHistory] = useState<OpportunityHistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadOpportunities = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getOpportunities(project.id);
      setOpportunities(data);
      if (data.length > 0 && !selectedOppId) {
        setSelectedOppId(data[0].id);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch opportunities.');
    } finally {
      setLoading(false);
    }
  }, [selectedOppId, project.id]);

  useEffect(() => {
    loadOpportunities();
  }, [loadOpportunities]);

  const selectedOpp = opportunities.find((o) => o.id === selectedOppId) || opportunities[0] || null;

  const loadEvidenceAndHistory = useCallback(async () => {
    if (!selectedOpp) return;
    const targetId = selectedOpp.lifecycle_opportunity_id || selectedOpp.id;
    try {
      const [evList, histList] = await Promise.all([
        api.getOpportunityEvidence(targetId, project.id).catch(() => []),
        api.getOpportunityHistory(targetId, project.id).catch(() => []),
      ]);
      setEvidence(evList);
      setHistory(histList);
    } catch (err: unknown) {
      console.warn('Failed to load evolution evidence/history:', err);
    }
  }, [selectedOpp, project.id]);

  useEffect(() => {
    loadEvidenceAndHistory();
  }, [loadEvidenceAndHistory]);

  const linkedPaperIds = new Set(evidence.map((e) => e.paper_id));

  if (loading) {
    return <LoadingState message="Loading opportunity evolution trajectories..." minHeight="400px" />;
  }

  if (error) {
    return (
      <div className="u-stack-lg">
        <ErrorBanner title="Could Not Load Evolution Data" message={error} onRetry={loadOpportunities} />
      </div>
    );
  }

  return (
    <div className="view-container animate-fade-in">
      <div className="view-header">
        <div>
          <div className="u-row" style={{ gap: '0.625rem', marginBottom: '0.25rem' }}>
            <span className="badge badge-primary">Stage 5 Pipeline</span>
            <span className="badge badge-default">Longitudinal Evolution</span>
          </div>
          <h1 className="view-title">Opportunity Evolution &amp; Validation</h1>
          <p className="view-subtitle">
            Classify post-cutoff evaluation literature against formulated opportunities using Ollama Qwen 3 (8B) to track state transitions.
          </p>
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => {
            loadOpportunities();
            loadEvidenceAndHistory();
          }}
        >
          <RefreshCw size={14} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Methodological limitations note */}
      <div className="callout callout-warning">
        <div className="callout-icon-col">
          <AlertTriangle size={20} />
        </div>
        <div className="callout-content">
          <div className="callout-title">Descriptive Trajectory Progression Note</div>
          <p className="callout-body">
            Stage 5 measures descriptive temporal evolution counts (relationships, transitions, status distribution). These values represent event volume and trajectory progression, not empirical accuracy, precision, or research validity without human-annotated gold standard data.
          </p>
        </div>
      </div>

      {opportunities.length === 0 ? (
        <div className="card u-stack-md" style={{ padding: '3rem', textAlign: 'center' }}>
          <div className="modal-icon-badge modal-icon-badge-primary" style={{ margin: '0 auto' }}>
            <Compass size={24} />
          </div>
          <h2 className="section-title">No Opportunities Available for Evolution</h2>
          <p className="text-secondary" style={{ maxWidth: '500px', margin: '0 auto' }}>
            Project <strong>{project.name}</strong> does not yet have formulated research opportunities. Discover research gaps and formulate an opportunity first.
          </p>
          <div className="u-row" style={{ justifyContent: 'center', marginTop: '1rem' }}>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => onNavigate(`#/projects/${project.id}/gaps`)}
            >
              <span>Go to Research Gaps</span>
              <ArrowRight size={14} />
            </button>
          </div>
        </div>
      ) : (
        <div className="u-stack-lg">
          {/* Opportunity Selection Banner */}
          <div className="card u-stack-sm">
            <span className="section-title" style={{ fontSize: '1rem' }}>Active Opportunity Target:</span>
            <div className="u-row" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
              {opportunities.map((opp) => (
                <button
                  key={opp.id}
                  type="button"
                  className={`btn ${opp.id === selectedOpp?.id ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => setSelectedOppId(opp.id)}
                  title={opp.proposed_research_direction || undefined}
                >
                  <Compass size={13} />
                  <span>{opp.proposed_research_direction?.slice(0, 45) || 'Opportunity'}...</span>
                  <Badge
                    label={opp.lifecycle_status || 'OPEN'}
                    variant={getLifecycleBadgeVariant(opp.lifecycle_status)}
                  />
                </button>
              ))}
            </div>

            {selectedOpp && (
              <div className="detail-box" style={{ marginTop: '0.5rem' }}>
                <div className="u-row" style={{ justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span className="font-semibold text-primary">
                    Direction: {selectedOpp.proposed_research_direction}
                  </span>
                  <Badge
                    label={`Status: ${selectedOpp.lifecycle_status || 'OPEN'}`}
                    variant={getLifecycleBadgeVariant(selectedOpp.lifecycle_status)}
                  />
                </div>
                <p className="text-secondary" style={{ fontSize: '0.8125rem' }}>
                  Problem: {selectedOpp.research_problem}
                </p>
              </div>
            )}
          </div>

          {/* Interactive Evolve Panel */}
          {selectedOpp && (
            <EvolvePanel
              projectId={project.id}
              lifecycleId={selectedOpp.lifecycle_opportunity_id || selectedOpp.id}
              opportunityCreatedAt={selectedOpp.created_at}
              linkedPaperIds={linkedPaperIds}
              onEvolved={() => {
                loadOpportunities();
                loadEvidenceAndHistory();
              }}
            />
          )}

          {/* Evolution Evidence and State Transition Audit Trail */}
          <div className="grid-2col">
            {/* Temporal Evidence Records */}
            <div className="card u-stack-md">
              <div className="u-row" style={{ justifyContent: 'space-between' }}>
                <h3 className="card-title">Temporal Evidence Records ({evidence.length})</h3>
                <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
                  Persisted Evidence
                </span>
              </div>

              {evidence.length === 0 ? (
                <p className="text-secondary" style={{ fontSize: '0.875rem' }}>
                  No evidence records linked to this opportunity yet. Select candidate papers in the panel above to classify relationships.
                </p>
              ) : (
                <div className="u-stack-sm">
                  {evidence.map((ev) => (
                    <div key={ev.id} className="detail-box">
                      <div className="u-row" style={{ justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                        <span className="font-semibold" style={{ fontSize: '0.8125rem' }}>
                          Paper: {ev.paper_title || ev.paper_id}
                        </span>
                        <Badge
                          label={ev.relationship}
                          variant={getRelationshipBadgeVariant(ev.relationship)}
                        />
                      </div>
                      <p className="text-secondary" style={{ fontSize: '0.75rem' }}>
                        {ev.supporting_text || 'Classified relationship via local LLM inference.'}
                      </p>
                      <div className="u-row text-tertiary" style={{ justifyContent: 'space-between', marginTop: '0.25rem', fontSize: '0.6875rem' }}>
                        <span>Confidence: {ev.confidence !== null ? `${Math.round(ev.confidence * 100)}%` : '—'}</span>
                        <span>Date: {ev.created_at ? ev.created_at.split('T')[0] : '—'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Lifecycle State Transitions History */}
            <div className="card u-stack-md">
              <div className="u-row" style={{ justifyContent: 'space-between' }}>
                <h3 className="card-title">Lifecycle State Transitions ({history.length})</h3>
                <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
                  Audit Log
                </span>
              </div>

              {history.length === 0 ? (
                <p className="text-secondary" style={{ fontSize: '0.875rem' }}>
                  No state transitions recorded. The opportunity remains in its initial status.
                </p>
              ) : (
                <div className="u-stack-sm">
                  {history.map((hist) => (
                    <div key={hist.id} className="detail-box">
                      <div className="u-row" style={{ justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                        <div className="u-row" style={{ gap: '0.375rem' }}>
                          <Badge
                            label={hist.previous_status}
                            variant={getLifecycleBadgeVariant(hist.previous_status)}
                          />
                          <span className="text-tertiary">&rarr;</span>
                          <Badge
                            label={hist.new_status}
                            variant={getLifecycleBadgeVariant(hist.new_status)}
                          />
                        </div>
                        <span className="badge badge-info" style={{ fontSize: '0.625rem' }}>
                          Trigger: {hist.trigger_type}
                        </span>
                      </div>
                      <span className="text-tertiary font-mono" style={{ fontSize: '0.6875rem' }}>
                        Timestamp: {hist.created_at}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
