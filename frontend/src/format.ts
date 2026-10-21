/**
 * Display helpers for backend-provided values.
 *
 * These never substitute invented defaults: a missing value is reported as
 * missing (null) so the caller can render the shared "Not recorded" marker.
 */

export const NOT_RECORDED_LABEL = 'Not recorded';

/** Returns a display string for a primitive value, or null when absent/unrenderable. */
export function displayValue(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') return value.trim() === '' ? null : value;
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : null;
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return null;
}


/** Formats a 0..1 fraction as a percentage, or null when absent. */
export function displayPercent(value: number | null | undefined): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return `${(value * 100).toFixed(0)}%`;
}
