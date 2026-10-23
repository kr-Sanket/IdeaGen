import React from 'react';
import type { IncrementalEvolutionResult } from '../../types';
import { Badge } from '../common/Badge';
import { getEvolutionHookBadgeVariant, getLifecycleBadgeVariant } from '../common/badgeVariants';

interface IngestEvolutionSummaryProps {
  status: string | null | undefined;
  result: IncrementalEvolutionResult | null | undefined;
}

function evolutionExplanation(status: string | null | undefined): string {
  switch (status) {
    case 'DISABLED':
      return 'Automatic evolution is disabled on the server (IDEAGEN_ENABLE_POST_INGESTION_EVOLUTION), so existing opportunities were not re-evaluated. You can evolve an opportunity manually from its detail page.';
    case 'SKIPPED':
      return 'Automatic evolution was enabled but skipped for this paper.';
    case 'SUCCESS':
      return 'Existing opportunities were re-evaluated against this paper.';
    case 'PARTIAL_SUCCESS':
      return 'Some opportunities were re-evaluated; others failed or were rejected (see counts).';
    case 'FAILED':
      return 'Automatic evolution failed. The paper itself was still stored.';
    default:
      return 'The server did not report an evolution status.';
  }
}

/** Post-ingestion evolution outcome exactly as reported by the ingest endpoint. */
export const IngestEvolutionSummary: React.FC<IngestEvolutionSummaryProps> = ({ status, result }) => {
  const outcomes = result?.outcomes ?? [];
  const changed = outcomes.filter((o) => o.state_changed);
  const reason = typeof result?.error === 'string' && result.error ? result.error : null;
  const hasCounts = result && typeof result.candidate_count === 'number';

  return (
    <div className="evolution-summary">
      <div className="u-row u-wrap">
        <Badge
          label={`Automatic evolution: ${status ?? 'Not reported'}`}
          variant={getEvolutionHookBadgeVariant(status)}
        />
        {result?.run_status && result.run_status !== status && (
          <span className="u-mono u-text-xs u-text-tertiary">run status: {result.run_status}</span>
        )}
      </div>
      <p className="u-text-xs u-text-secondary">{evolutionExplanation(status)}</p>
      {reason && <p className="u-text-xs u-mono u-text-secondary">Reason: {reason}</p>}

      {hasCounts && (
        <div className="count-row">
          <span>Candidates: {result.candidate_count}</span>
          <span>Evaluated: {result.evaluated_count ?? 0}</span>
          <span>Skipped: {result.skipped_count ?? 0}</span>
          <span>Grounding rejected: {result.grounding_rejected_count ?? 0}</span>
          <span>Failed: {result.failed_count ?? 0}</span>
        </div>
      )}

      {changed.length > 0 && (
        <ul className="evolution-change-list">
          {changed.map((o) => (
            <li key={`${o.opportunity_id}-${o.evidence_id ?? ''}`} className="u-row u-wrap u-text-xs">
              <a className="btn-link" href={`#/opportunities/${o.opportunity_id}`}>
                Lifecycle {o.opportunity_id.slice(0, 8)}…
              </a>
              {o.relationship && <span className="u-mono">{o.relationship}</span>}
              <Badge label={o.initial_status ?? '?'} variant={getLifecycleBadgeVariant(o.initial_status)} />
              <span>→</span>
              <Badge label={o.final_status ?? '?'} variant={getLifecycleBadgeVariant(o.final_status)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
