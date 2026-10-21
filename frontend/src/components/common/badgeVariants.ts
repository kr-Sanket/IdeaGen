/** Badge variant mapping helpers (kept separate from Badge.tsx for fast refresh). */

export type BadgeVariant =
  | 'lifecycle-open'
  | 'lifecycle-strengthened'
  | 'lifecycle-partially-addressed'
  | 'lifecycle-resolved'
  | 'relationship-supports'
  | 'relationship-addresses'
  | 'relationship-contradicts'
  | 'relationship-extends'
  | 'relationship-unrelated'
  | 'screening-selected'
  | 'screening-excluded'
  | 'temporal-baseline'
  | 'temporal-later'
  | 'statement-limitation'
  | 'statement-future_work'
  | 'statement-methodology'
  | 'statement-finding'
  | 'statement-observation'
  | 'neutral'
  | 'model-generated';

/**
 * Maps lifecycle states (backend/ideagen/opportunity/models.py OpportunityStatus)
 * to badge variants. Unknown or missing values render as neutral.
 */
export function getLifecycleBadgeVariant(status: string | null | undefined): BadgeVariant {
  switch (status) {
    case 'OPEN':
      return 'lifecycle-open';
    case 'STRENGTHENED':
      return 'lifecycle-strengthened';
    case 'PARTIALLY_ADDRESSED':
      return 'lifecycle-partially-addressed';
    case 'RESOLVED':
      return 'lifecycle-resolved';
    default:
      return 'neutral';
  }
}

/**
 * Maps evidence relationship types (backend OpportunityRelationship) to badge variants.
 */
export function getRelationshipBadgeVariant(relationship: string | null | undefined): BadgeVariant {
  switch (relationship) {
    case 'SUPPORTS':
      return 'relationship-supports';
    case 'ADDRESSES':
      return 'relationship-addresses';
    case 'CONTRADICTS':
      return 'relationship-contradicts';
    case 'EXTENDS':
      return 'relationship-extends';
    case 'UNRELATED':
      return 'relationship-unrelated';
    default:
      return 'neutral';
  }
}

/**
 * Maps relevance screening decisions (PaperScreeningResult.decision) to badge variants.
 */
export function getScreeningDecisionBadgeVariant(decision: string | null | undefined): BadgeVariant {
  if (decision?.startsWith('SELECTED')) return 'screening-selected';
  if (decision?.startsWith('EXCLUDED')) return 'screening-excluded';
  return 'neutral';
}

/**
 * Maps temporal_role (resolve_paper_temporal_role) to a badge variant.
 * UNDATED and DATED (no cutoff configured) render as neutral.
 */
export function getTemporalRoleBadgeVariant(role: string | null | undefined): BadgeVariant {
  if (role === 'BASELINE') return 'temporal-baseline';
  if (role === 'LATER_EVALUATION') return 'temporal-later';
  return 'neutral';
}

/**
 * Human-readable label derived only from temporal_role.
 */
export function getTemporalRoleLabel(role: string | null | undefined, cutoff?: string | null): string {
  switch (role) {
    case 'BASELINE':
      return cutoff ? `Baseline (≤ ${cutoff})` : 'Baseline (≤ cutoff)';
    case 'LATER_EVALUATION':
      return cutoff ? `Later evaluation (> ${cutoff})` : 'Later evaluation (> cutoff)';
    case 'UNDATED':
      return 'Undated';
    case 'DATED':
      return 'Dated (no cutoff configured)';
    default:
      return 'Temporal role not recorded';
  }
}

export function getStatementTypeBadgeVariant(type: string | null | undefined): BadgeVariant {
  switch (type?.toUpperCase()) {
    case 'LIMITATION':
      return 'statement-limitation';
    case 'FUTURE_WORK':
      return 'statement-future_work';
    case 'METHOD':
    case 'METHODOLOGY':
      return 'statement-methodology';
    case 'FINDING':
      return 'statement-finding';
    default:
      return 'statement-observation';
  }
}
