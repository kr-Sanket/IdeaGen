import React, { type ReactNode } from 'react';
import { Search } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: ReactNode;
  action?: ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No records found',
  description = 'No results match the current filter criteria or run database.',
  icon,
  action,
}) => {
  return (
    <div className="empty-state-card">
      <div className="empty-state-icon-wrapper">
        {icon || <Search size={24} />}
      </div>
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-description">{description}</p>
      {action && <div className="empty-state-action">{action}</div>}
    </div>
  );
};
