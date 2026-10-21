import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorBannerProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({
  title = 'API Request Failed',
  message,
  onRetry,
}) => {
  return (
    <div className="error-banner">
      <div className="error-banner-content">
        <AlertCircle className="error-banner-icon" />
        <div>
          <h4 className="error-banner-title">{title}</h4>
          <p className="error-banner-message">{message}</p>
        </div>
      </div>
      {onRetry && (
        <button onClick={onRetry} className="error-banner-retry-btn">
          <RefreshCw size={14} />
          <span>Retry</span>
        </button>
      )}
    </div>
  );
};
