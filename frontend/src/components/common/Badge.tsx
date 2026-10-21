import React from 'react';
import type { BadgeVariant } from './badgeVariants';

interface BadgeProps {
  label: string;
  variant?: BadgeVariant;
  className?: string;
  title?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  label,
  variant = 'neutral',
  className = '',
  title,
}) => {
  return (
    <span
      className={`app-badge badge-${variant} ${className}`}
      title={title}
    >
      {label}
    </span>
  );
};
