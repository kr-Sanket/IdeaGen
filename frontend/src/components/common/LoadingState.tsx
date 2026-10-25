import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  submessage?: string;
  minHeight?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading experiment data...',
  submessage = 'Querying the IdeaGen API',
  minHeight = '300px',
}) => {
  return (
    <div className="loading-state-container" style={{ minHeight }}>
      <div className="loading-spinner-wrapper">
        <Loader2 className="loading-spinner" />
      </div>
      <p className="loading-message">{message}</p>
      {submessage && <p className="loading-submessage">{submessage}</p>}
    </div>
  );
};
