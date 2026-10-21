import React, { useEffect, useState } from 'react';
import { Sparkles, Compass, BookOpen, Activity } from 'lucide-react';
import { api } from '../api/client';
import type { RouteName } from '../router';

interface NavbarProps {
  currentRoute: RouteName;
  onNavigate: (route: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentRoute, onNavigate }) => {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const ok = await api.checkBackendConnection();
      if (mounted) setBackendOnline(ok);
    };
    check();
    const interval = setInterval(check, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems = [
    {
      id: 'overview',
      path: '#/overview',
      label: 'Run Overview',
      icon: <Activity size={16} />,
      active: currentRoute === 'overview',
    },
    {
      id: 'opportunities',
      path: '#/opportunities',
      label: 'Opportunities Explorer',
      icon: <Compass size={16} />,
      active: currentRoute === 'opportunities' || currentRoute === 'opportunity-detail',
    },
    {
      id: 'corpus',
      path: '#/corpus',
      label: 'Corpus & Statements',
      icon: <BookOpen size={16} />,
      active: currentRoute === 'corpus',
    },
  ];

  return (
    <header className="app-navbar">
      <div className="navbar-container">
        {/* Brand identity */}
        <div
          className="navbar-brand"
          onClick={() => onNavigate('#/overview')}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter') onNavigate('#/overview'); }}
        >
          <div className="navbar-brand-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <div className="u-row">
              <span className="navbar-brand-name">IdeaGen</span>
              <span className="navbar-brand-badge">Read-Only Explorer</span>
            </div>
            <span className="navbar-brand-subtitle">
              Proof-of-concept research experiment archive
            </span>
          </div>
        </div>

        {/* Primary View Navigation */}
        <nav className="navbar-nav" aria-label="Main Navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onNavigate(item.path)}
              className={`navbar-tab-btn ${item.active ? 'active' : ''}`}
              aria-current={item.active ? 'page' : undefined}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        {/* Honest Backend Connectivity Indicator */}
        <div className="navbar-status-wrapper">
          {backendOnline === null ? (
            <div className="status-indicator status-checking">
              <span className="status-dot animate-pulse" />
              <span>Checking API...</span>
            </div>
          ) : backendOnline ? (
            <div className="status-indicator status-online" title="Backend FastAPI server is responding">
              <span className="status-dot status-dot-online" />
              <span>API Connected</span>
            </div>
          ) : (
            <div className="status-indicator status-offline" title="Backend FastAPI server is unreachable or offline">
              <span className="status-dot status-dot-offline" />
              <span>API Offline</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
