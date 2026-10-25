import React, { Component, type ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('IdeaGen Explorer encountered an unhandled error:', error, errorInfo);
  }

  handleReload = (): void => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="error-boundary-container">
          <div className="error-boundary-card">
            <div className="error-boundary-icon-wrapper">
              <AlertTriangle className="error-boundary-icon" />
            </div>
            <h2 className="error-boundary-title">Application Render Error</h2>
            <p className="error-boundary-description">
              The IdeaGen Research Explorer encountered an unexpected client error while rendering this view.
            </p>
            {this.state.error?.message && (
              <pre className="error-boundary-trace">
                {this.state.error.message}
              </pre>
            )}
            <button
              onClick={this.handleReload}
              className="btn btn-primary"
            >
              <RefreshCw size={16} />
              <span>Reload Explorer</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
