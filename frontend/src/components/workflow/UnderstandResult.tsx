import React, { type ReactNode } from 'react';
import type { ResearchKnowledge } from '../../types';
import { Badge } from '../common/Badge';

interface UnderstandResultProps {
  knowledge: ResearchKnowledge;
  action?: ReactNode;
}

/** Result of POST /papers/{id}/understand, reported as returned by the backend. */
export const UnderstandResult: React.FC<UnderstandResultProps> = ({ knowledge, action }) => {
  const succeeded = knowledge.extraction_status === 'SUCCESS';
  const variant = succeeded && knowledge.statement_count > 0 ? 'status-positive' : succeeded ? 'status-warning' : 'status-negative';

  return (
    <div className={`action-result ${variant === 'status-positive' ? 'is-success' : variant === 'status-warning' ? 'is-warning' : 'is-error'}`} role="status">
      <div className="u-row u-wrap">
        <Badge label={`Extraction: ${knowledge.extraction_status}`} variant={variant} />
        <span className="u-text-xs u-text-tertiary u-mono">Model: {knowledge.model_name}</span>
      </div>
      <p className="u-text-sm">
        {knowledge.statement_count} research statement{knowledge.statement_count === 1 ? '' : 's'} stored
        {knowledge.statement_count > 0
          ? ` (${knowledge.limitations.length} limitation${knowledge.limitations.length === 1 ? '' : 's'}, ${knowledge.future_work.length} future-work item${knowledge.future_work.length === 1 ? '' : 's'} summarized).`
          : '. Gap discovery needs statements, so this paper will not contribute to gaps.'}
      </p>
      {knowledge.error_message && (
        <p className="u-text-xs u-mono u-text-secondary action-result-detail">{knowledge.error_message}</p>
      )}
      {action && <div className="u-row u-wrap action-result-actions">{action}</div>}
    </div>
  );
};
