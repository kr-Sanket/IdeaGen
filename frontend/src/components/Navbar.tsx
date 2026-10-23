import React, { useEffect, useState, useRef } from 'react';
import {
  Layers,
  Compass,
  BookOpen,
  LayoutDashboard,
  Settings,
  Lightbulb,
  GitBranch,
  History,
  ChevronDown,
  Check,
  FolderPlus,
  Play
} from 'lucide-react';
import { api } from '../api/client';
import type { RouteName } from '../router';
import type { BackendHealth, ResearchProject } from '../types';
import { syncProjectsWithServer } from '../projects/projectStore';

interface NavbarProps {
  currentRoute: RouteName;
  activeProject: ResearchProject;
  projects: ResearchProject[];
  onSelectProject: (projectId: string) => void;
  onOpenCreateProject: () => void;
  onOpenNewRunModal: () => void;
  onNavigate: (route: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentRoute,
  activeProject,
  projects,
  onSelectProject,
  onOpenCreateProject,
  onOpenNewRunModal,
  onNavigate,
}) => {
  const [backendHealth, setBackendHealth] = useState<BackendHealth | null>(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Poll faster while the API is not healthy so recovery shows without a reload, and re-check
  // immediately when the window regains focus or the browser comes back online.
  useEffect(() => {
    let mounted = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let previous: BackendHealth | null = null;
    const check = async () => {
      if (timer) clearTimeout(timer);
      const health = await api.checkBackendHealth();
      if (!mounted) return;
      setBackendHealth(health);
      if (health === 'online' && previous !== 'online') {
        // Register any locally created projects the backend has not seen yet.
        void syncProjectsWithServer();
      }
      previous = health;
      timer = setTimeout(check, health === 'online' ? 15000 : 3000);
    };
    void check();
    const onWake = () => void check();
    window.addEventListener('focus', onWake);
    window.addEventListener('online', onWake);
    return () => {
      mounted = false;
      if (timer) clearTimeout(timer);
      window.removeEventListener('focus', onWake);
      window.removeEventListener('online', onWake);
    };
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const globalNavItems = [
    {
      id: 'dashboard',
      path: '#/dashboard',
      label: 'Dashboard',
      icon: <LayoutDashboard size={15} />,
      active: currentRoute === 'dashboard',
    },
    {
      id: 'workspaces',
      path: '#/workspaces',
      label: 'Workspaces',
      icon: <Layers size={15} />,
      active: currentRoute === 'workspaces',
    },
    {
      id: 'settings',
      path: '#/settings',
      label: 'Settings',
      icon: <Settings size={15} />,
      active: currentRoute === 'settings',
    },
  ];

  const projectNavItems = [
    {
      id: 'overview',
      path: `#/projects/${activeProject.id}/overview`,
      label: 'Overview',
      icon: <LayoutDashboard size={14} />,
      active: currentRoute === 'project-overview' || currentRoute === 'overview',
    },
    {
      id: 'papers',
      path: `#/projects/${activeProject.id}/papers`,
      label: 'Papers',
      icon: <BookOpen size={14} />,
      active: currentRoute === 'project-papers' || currentRoute === 'search' || currentRoute === 'corpus',
    },
    {
      id: 'gaps',
      path: `#/projects/${activeProject.id}/gaps`,
      label: 'Research Gaps',
      icon: <Lightbulb size={14} />,
      active: currentRoute === 'project-gaps' || currentRoute === 'gaps',
    },
    {
      id: 'opportunities',
      path: `#/projects/${activeProject.id}/opportunities`,
      label: 'Opportunities',
      icon: <Compass size={14} />,
      active: currentRoute === 'project-opportunities' || currentRoute === 'project-opportunity-detail' || currentRoute === 'opportunities' || currentRoute === 'opportunity-detail',
    },
    {
      id: 'evolution',
      path: `#/projects/${activeProject.id}/evolution`,
      label: 'Evolution',
      icon: <GitBranch size={14} />,
      active: currentRoute === 'project-evolution' || currentRoute === 'evolution',
    },
    {
      id: 'history',
      path: `#/projects/${activeProject.id}/history`,
      label: 'Run History',
      icon: <History size={14} />,
      active: currentRoute === 'project-history' || currentRoute === 'history',
    },
  ];

  const handleSelectProj = (id: string) => {
    setIsDropdownOpen(false);
    onSelectProject(id);
    onNavigate(`#/projects/${id}/overview`);
  };

  return (
    <header className="app-navbar">
      {/* ── Top Tier: Global Shell Navigation ── */}
      <div className="navbar-container">
        {/* Brand identity */}
        <button
          type="button"
          className="navbar-brand"
          onClick={() => onNavigate('#/dashboard')}
          aria-label="IdeaGen home: Dashboard"
        >
          <img
            src="/ideagen logo 1.jpeg"
            alt="IdeaGen Logo"
            className="navbar-logo-img"
          />
          <div className="navbar-brand-text">
            <div className="navbar-brand-row">
              <span className="navbar-brand-name">IdeaGen</span>
              <span className="navbar-brand-badge">Research Lab</span>
            </div>
            <span className="navbar-brand-subtitle">
              Evidence-Grounded Opportunity Discovery
            </span>
          </div>
        </button>

        {/* Global Navigation Tabs */}
        <nav className="navbar-nav" aria-label="Global Navigation">
          {globalNavItems.map((item) => (
            <button
              type="button"
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

        {/* Project Switcher Dropdown & Connectivity */}
        <div className="u-row" style={{ gap: '0.875rem' }}>
          {/* Active Project Switcher */}
          <div className="project-switcher-container" ref={dropdownRef}>
            <button
              type="button"
              className="project-switcher-btn"
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              aria-expanded={isDropdownOpen}
              aria-haspopup="true"
              id="project-switcher-trigger"
            >
              <div className="switcher-label-col">
                <span className="switcher-hint">Current Project</span>
                <span className="switcher-active-name" title={activeProject.name}>
                  {activeProject.name}
                </span>
              </div>
              <ChevronDown size={14} className={`switcher-arrow ${isDropdownOpen ? 'open' : ''}`} />
            </button>

            {isDropdownOpen && (
              <div className="project-dropdown-menu animate-fade-in" role="menu">
                <div className="dropdown-section-label">Your Projects</div>
                <div className="dropdown-projects-list">
                  {projects.map((proj) => {
                    const isSelected = proj.id === activeProject.id;
                    return (
                      <button
                        key={proj.id}
                        type="button"
                        className={`dropdown-project-item ${isSelected ? 'selected' : ''}`}
                        onClick={() => handleSelectProj(proj.id)}
                        role="menuitem"
                      >
                        <div className="dropdown-item-info">
                          <div className="u-row" style={{ gap: '0.375rem' }}>
                            <span className="dropdown-item-title">{proj.name}</span>
                            {proj.isDefaultBenchmark && (
                              <span className="badge badge-primary" style={{ fontSize: '0.5625rem' }}>
                                Benchmark
                              </span>
                            )}
                          </div>
                          <span className="dropdown-item-topic">{proj.topic}</span>
                        </div>
                        {isSelected && <Check size={14} className="text-primary" />}
                      </button>
                    );
                  })}
                </div>

                <div className="dropdown-divider" />

                <button
                  type="button"
                  className="dropdown-create-btn"
                  onClick={() => {
                    setIsDropdownOpen(false);
                    onOpenCreateProject();
                  }}
                  role="menuitem"
                >
                  <FolderPlus size={14} />
                  <span>+ Create New Project</span>
                </button>
              </div>
            )}
          </div>

          {/* Backend Connectivity Status */}
          <div className="navbar-status-wrapper">
            {backendHealth === null ? (
              <div className="status-indicator status-checking" title="Connecting to the IdeaGen API...">
                <span className="status-dot animate-pulse" />
                <span className="status-label">Checking</span>
              </div>
            ) : backendHealth === 'online' ? (
              <div className="status-indicator status-online" title="The IdeaGen API and its database are responding.">
                <span className="status-dot status-dot-online" />
                <span className="status-label">API Connected</span>
              </div>
            ) : backendHealth === 'degraded' ? (
              <div className="status-indicator status-offline" title="The API is running but reports a database connection error.">
                <span className="status-dot status-dot-offline" />
                <span className="status-label">Database Error</span>
              </div>
            ) : (
              <div
                className="status-indicator status-offline"
                title="No response from the IdeaGen API through the /api dev proxy. Check that the backend is running and that the frontend process may open local connections. Retrying every few seconds."
                role="status"
              >
                <span className="status-dot status-dot-offline" />
                <span className="status-label">API Offline</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Second Tier: Project Context & Sub-navigation ── */}
      <div className="navbar-project-bar">
        <div className="navbar-project-container">
          {/* Active project context pill */}
          <div className="project-context-badge">
            <span className="project-context-label">Project:</span>
            <strong className="project-context-name" title={activeProject.name}>
              {activeProject.name}
            </strong>
            <span className="badge badge-default" style={{ fontSize: '0.625rem' }}>
              Cutoff: {activeProject.temporalCutoff || 'None'}
            </span>
          </div>

          {/* Project-specific navigation items */}
          <nav className="project-nav-items" aria-label="Project Sections">
            {projectNavItems.map((item) => (
              <button
                type="button"
                key={item.id}
                onClick={() => onNavigate(item.path)}
                className={`project-tab-btn ${item.active ? 'active' : ''}`}
                aria-current={item.active ? 'page' : undefined}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            ))}
          </nav>

          {/* Prominent New Research Run Action */}
          <button
            type="button"
            className="btn btn-primary btn-sm btn-new-run"
            onClick={onOpenNewRunModal}
            id="nav-btn-new-run"
            title="Start or configure a new research run"
          >
            <Play size={13} />
            <span>New Research Run</span>
          </button>
        </div>
      </div>
    </header>
  );
};
