import React from 'react';
import { ArrowRight } from 'lucide-react';
import type { OpportunityReport } from '../../types';
import { Badge } from '../common/Badge';
import { getGenerationStatusBadgeVariant, getLifecycleBadgeVariant } from '../common/badgeVariants';
import { RecordedValue } from '../common/NotRecorded';
import { displayPercent } from '../../format';

interface GenerationResultProps {
  report: OpportunityReport;
  onOpen: (generationId: string) => void;
}

function statusExplanation(report: OpportunityReport): string {
  const n = report.refinement_iteration_count;
  switch (report.status) {
    case 'VALIDATED':
      return 'Accepted without refinement: the prior-art validator did not report high overlap with statements retrieved from the corpus.';
    case 'REFINED':
      return `Accepted after ${n} refinement iteration${n === 1 ? '' : 's'}: the first draft overlapped heavily with corpus prior art; the revised proposal no longer did, according to the validator.`;
    case 'REJECTED_INSUFFICIENTLY_DIFFERENTIATED':
    case 'INSUFFICIENTLY_DIFFERENTIATED':
      return `Rejected: after ${n} refinement iteration${n === 1 ? '' : 's'} the proposal still overlapped heavily with existing work in the corpus (or the refiner rejected it). The record is saved for audit but is not tracked in the opportunity lifecycle.`;
    default:
      return `The pipeline returned status ${report.status}.`;
  }
}

/** Outcome of POST /gaps/{id}/opportunity, using only fields returned by the backend. */
export const GenerationResult: React.FC<GenerationResultProps> = ({ report, onOpen }) => {
  const accepted = report.status === 'VALIDATED' || report.status === 'REFINED';
  const novelty = report.novelty_assessment;

  return (
    <div className={`action-result ${accepted ? 'is-success' : 'is-warning'}`} role="status">
      <div className="u-row u-wrap">
        <Badge label={`Generation: ${report.status}`} variant={getGenerationStatusBadgeVariant(report.status)} />
        {report.lifecycle_status ? (
          <Badge label={`Lifecycle: ${report.lifecycle_status}`} variant={getLifecycleBadgeVariant(report.lifecycle_status)} />
        ) : (
          <Badge label="Lifecycle: not tracked" variant="neutral" />
        )}
        <Badge label="Model-Generated" variant="model-generated" />
      </div>

      <p className="u-text-sm">{statusExplanation(report)}</p>

      <h4 className="action-result-heading">{report.proposed_research_direction}</h4>

      <div className="u-stack-sm">
        <div>
          <span className="label-caps">Prior-art validator reasoning</span>
          <p className="prose-small"><RecordedValue value={novelty?.reasoning} /></p>
        </div>
        <div>
          <span className="label-caps">Differentiation</span>
          <p className="prose-small"><RecordedValue value={novelty?.differentiation ?? report.differentiation} /></p>
        </div>
        <div className="count-row">
          <span>Model-reported confidence: <RecordedValue value={displayPercent(report.confidence)} /></span>
          <span>Validator confidence: <RecordedValue value={displayPercent(novelty?.confidence)} /></span>
          <span>Supporting statements: {report.supporting_evidence.length}</span>
          <span>Supporting papers: {report.supporting_papers.length}</span>
        </div>
      </div>

      <dl className="id-list">
        <div>
          <dt>Generation record ID</dt>
          <dd className="u-mono">{report.opportunity_id}</dd>
        </div>
        <div>
          <dt>Lifecycle opportunity ID</dt>
          <dd className="u-mono">{report.lifecycle_opportunity_id ?? 'None (not bridged)'}</dd>
        </div>
      </dl>

      <div className="u-row u-wrap action-result-actions">
        <button type="button" className="btn btn-primary btn-sm" onClick={() => onOpen(report.opportunity_id)}>
          <span>Open full report</span>
          <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
};
