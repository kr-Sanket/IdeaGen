import { useState, useEffect } from 'react';
import { useRoute } from './router';
import { useProjectContext } from './projects/projectStore';
import { Navbar } from './components/Navbar';
import { DashboardView } from './components/views/DashboardView';
import { WorkspacesView } from './components/views/WorkspacesView';
import { SettingsView } from './components/views/SettingsView';
import { ProjectOverviewView } from './components/views/ProjectOverviewView';
import { ProjectPapersView } from './components/views/ProjectPapersView';
import { ResearchGapsView } from './components/views/ResearchGapsView';
import { OpportunitiesExplorerView } from './components/views/OpportunitiesExplorerView';
import { OpportunityDetailView } from './components/views/OpportunityDetailView';
import { EvolutionView } from './components/views/EvolutionView';
import { RunHistoryView } from './components/views/RunHistoryView';
import { CreateProjectModal } from './components/views/CreateProjectModal';
import { NewRunModal } from './components/views/NewRunModal';
import { RunProgressModal } from './components/workflow/RunProgressModal';
import { EvidenceDrawer } from './components/EvidenceDrawer';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import { useRunMonitor } from './workflow/runMonitor';
import { Lightbulb, CheckCircle2, AlertTriangle, AlertCircle, X } from 'lucide-react';
import './App.css';

export function App() {
  const { route, navigate } = useRoute();
  const {
    projects,
    activeProject,
    activeProjectId,
    setActiveProject,
    createProject,
    deleteProject,
  } = useProjectContext();

  const {
    notification,
    dismissNotification,
    modalState,
    closeRunProgress,
    openRunProgress,
  } = useRunMonitor();

  const [activeStatementId, setActiveStatementId] = useState<string | null>(null);
  const [isCreateProjectOpen, setIsCreateProjectOpen] = useState<boolean>(false);
  const [isNewRunOpen, setIsNewRunOpen] = useState<boolean>(false);

  // Sync route projectId if provided in URL (e.g. #/projects/:id/...)
  useEffect(() => {
    if (route.params.projectId && route.params.projectId !== activeProjectId) {
      const match = projects.find((p) => p.id === route.params.projectId);
      if (match) {
        setActiveProject(match.id);
      }
    }
  }, [route.params.projectId, activeProjectId, projects, setActiveProject]);

  // Deep-link check for statement_id in query params
  useEffect(() => {
    if (route.params.statement_id) {
      setActiveStatementId(route.params.statement_id);
    }
  }, [route.params.statement_id]);

  const handleInspectEvidence = (statementId: string) => {
    setActiveStatementId(statementId);
  };

  const handleCloseEvidence = () => {
    setActiveStatementId(null);
  };

  const handleSelectProject = (projectId: string) => {
    setActiveProject(projectId);
  };

  return (
    <ErrorBoundary>
      <div className="app-shell">
        {/* Navigation Bar (Two-tier global + project context) */}
        <Navbar
          currentRoute={route.name}
          activeProject={activeProject}
          projects={projects}
          onSelectProject={handleSelectProject}
          onOpenCreateProject={() => setIsCreateProjectOpen(true)}
          onOpenNewRunModal={() => setIsNewRunOpen(true)}
          onNavigate={navigate}
        />

        {/* Persistent Run Completion & Review Notification Banner */}
        {notification && (
          <div
            className={`run-notification-banner banner-${notification.variant} animate-fade-in`}
            role="status"
            aria-live="polite"
            id="run-completion-notification"
          >
            <div className="banner-content">
              <div className="banner-icon-badge">
                {notification.status === 'AWAITING_REVIEW' && <Lightbulb size={16} className="text-warning" />}
                {notification.status === 'COMPLETED' && <CheckCircle2 size={16} className="text-success" />}
                {notification.status === 'PARTIAL_SUCCESS' && <AlertTriangle size={16} className="text-warning" />}
                {(notification.status === 'FAILED' || notification.status === 'INTERRUPTED') && (
                  <AlertCircle size={16} className="text-danger" />
                )}
              </div>
              <div className="banner-text">
                <strong className="banner-title">{notification.title}: </strong>
                <span className="banner-message">{notification.message}</span>
                {notification.projectName && (
                  <span className="badge badge-default banner-project-pill">
                    Project: {notification.projectName}
                  </span>
                )}
              </div>
            </div>
            <div className="banner-actions">
              {notification.status === 'AWAITING_REVIEW' && (
                <button
                  type="button"
                  className="btn btn-primary btn-xs"
                  onClick={() => {
                    if (notification.projectId !== activeProjectId) {
                      setActiveProject(notification.projectId);
                    }
                    navigate(`#/projects/${notification.projectId}/gaps`);
                    dismissNotification();
                  }}
                  id="btn-banner-inspect-gaps"
                >
                  Inspect Gaps &rarr;
                </button>
              )}
              <button
                type="button"
                className="btn btn-secondary btn-xs"
                onClick={() => {
                  if (notification.projectId !== activeProjectId) {
                    setActiveProject(notification.projectId);
                  }
                  openRunProgress(notification.runId, notification.projectId);
                }}
                id="btn-banner-view-run"
              >
                View Run &amp; Logs
              </button>
              <button
                type="button"
                className="btn-icon btn-xs"
                onClick={dismissNotification}
                aria-label="Dismiss notification"
                id="btn-banner-dismiss"
              >
                <X size={15} />
              </button>
            </div>
          </div>
        )}

        {/* Primary Page Content */}
        <main className="app-main-content">
          {/* Global Views */}
          {route.name === 'dashboard' && (
            <DashboardView
              projects={projects}
              activeProjectId={activeProjectId}
              onSelectProject={handleSelectProject}
              onCreateProject={createProject}
              onDeleteProject={deleteProject}
              onNavigate={navigate}
            />
          )}

          {route.name === 'workspaces' && (
            <WorkspacesView
              projects={projects}
              activeProjectId={activeProjectId}
              onSelectProject={handleSelectProject}
              onNavigate={navigate}
            />
          )}

          {route.name === 'settings' && (
            <SettingsView />
          )}

          {/* Project-Scoped Views */}
          {(route.name === 'project-overview' || route.name === 'overview') && (
            <ProjectOverviewView
              project={activeProject}
              onNavigate={navigate}
              onDeleteProject={deleteProject}
            />
          )}

          {(route.name === 'project-papers' || route.name === 'search' || route.name === 'corpus') && (
            <ProjectPapersView
              project={activeProject}
              initialTab={
                route.params.tab === 'corpus' || route.name === 'corpus'
                  ? 'corpus'
                  : 'search'
              }
              initialPaperId={route.params.paper_id || null}
              onNavigate={navigate}
              onInspectEvidence={handleInspectEvidence}
            />
          )}

          {(route.name === 'project-gaps' || route.name === 'gaps') && (
            <ResearchGapsView
              project={activeProject}
              initialGapId={route.params.gap_id || null}
              onNavigate={navigate}
              onInspectEvidence={handleInspectEvidence}
            />
          )}

          {(route.name === 'project-opportunities' || route.name === 'opportunities') && (
            <OpportunitiesExplorerView
              project={activeProject}
              onNavigate={navigate}
            />
          )}

          {(route.name === 'project-opportunity-detail' || route.name === 'opportunity-detail') && (
            <OpportunityDetailView
              opportunityId={route.params.id || ''}
              projectId={activeProjectId}
              onNavigate={navigate}
              onInspectEvidence={handleInspectEvidence}
            />
          )}

          {(route.name === 'project-evolution' || route.name === 'evolution') && (
            <EvolutionView
              project={activeProject}
              onNavigate={navigate}
            />
          )}

          {(route.name === 'project-history' || route.name === 'history') && (
            <RunHistoryView
              project={activeProject}
              onNavigate={navigate}
            />
          )}
        </main>

        {/* Global Modals */}
        <CreateProjectModal
          isOpen={isCreateProjectOpen}
          onClose={() => setIsCreateProjectOpen(false)}
          onCreate={createProject}
          onSelectProject={(id) => {
            handleSelectProject(id);
            navigate(`#/projects/${id}/overview`);
          }}
        />

        <NewRunModal
          isOpen={isNewRunOpen}
          onClose={() => setIsNewRunOpen(false)}
          project={activeProject}
          onNavigate={navigate}
        />

        {/* Global Authoritative Run Progress Monitor Modal */}
        <RunProgressModal
          isOpen={modalState.isOpen}
          onClose={closeRunProgress}
          runId={modalState.runId}
          projectId={modalState.projectId || activeProjectId}
          onNavigate={navigate}
        />

        {/* Statement Source Provenance Drawer */}
        <EvidenceDrawer
          statementId={activeStatementId}
          onClose={handleCloseEvidence}
        />

        {/* Application Footer */}
        <footer className="app-footer">
          <div className="app-footer-container">
            <span className="app-footer-brand">
              IdeaGen — Evidence-Grounded Research Discovery &amp; Opportunity Evolution Archive
            </span>
            <span className="app-footer-tagline">
              Personal Workspace · Model outputs are grounded in stored passages and verified locally
            </span>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

export default App;
