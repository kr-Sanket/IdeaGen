import React from 'react';
import { NOT_RECORDED_LABEL, displayValue } from '../../format';

interface NotRecordedProps {
  label?: string;
}

/** Shared muted marker for values the backend did not provide. */
export const NotRecorded: React.FC<NotRecordedProps> = ({ label = NOT_RECORDED_LABEL }) => (
  <span className="not-recorded">{label}</span>
);

interface RecordedValueProps {
  value: unknown;
  suffix?: string;
  missingLabel?: string;
}

/** Renders a backend value as-is (plus optional unit suffix), or "Not recorded". */
export const RecordedValue: React.FC<RecordedValueProps> = ({ value, suffix, missingLabel }) => {
  const text = displayValue(value);
  if (text === null) return <NotRecorded label={missingLabel} />;
  return <>{suffix ? `${text} ${suffix}` : text}</>;
};
