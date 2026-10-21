import { useState, useEffect } from 'react';
import { useRoute } from './router';
import { Navbar } from './components/Navbar';
import { RunOverviewView } from './components/views/RunOverviewView';
import { OpportunitiesExplorerView } from './components/views/OpportunitiesExplorerView';
import { OpportunityDetailView } from './components/views/OpportunityDetailView';
import { CorpusInspectorView } from './components/views/CorpusInspectorView';
import { EvidenceDrawer } from './components/EvidenceDrawer';
import { ErrorBoundary } from './components/common/ErrorBoundary';
import './App.css';

export function App() {
  const { route, navigate } = useRoute();
  const [activeStatementId, setActiveStatementId] = useState<string | null>(null);

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

  return (
    <ErrorBoundary>
      <div className="app-shell">
        {/* Navigation Bar */}
        <Navbar
          currentRoute={route.name}
          onNavigate={navigate}
        />

        {/* Primary Page Content */}
        <main className="app-main-content">
          {route.name === 'overview' && (
            <RunOverviewView onNavigate={navigate} />
          )}

          {route.name === 'opportunities' && (
            <OpportunitiesExplorerView onNavigate={navigate} />
          )}

          {route.name === 'opportunity-detail' && (
            <OpportunityDetailView
              opportunityId={route.params.id || ''}
              onNavigate={navigate}
              onInspectEvidence={handleInspectEvidence}
            />
          )}

          {route.name === 'corpus' && (
            <CorpusInspectorView
              initialPaperId={route.params.paper_id || null}
              onInspectEvidence={handleInspectEvidence}
            />
          )}
        </main>

        {/* Statement Source Provenance Drawer */}
        <EvidenceDrawer
          statementId={activeStatementId}
          onClose={handleCloseEvidence}
        />

        {/* Application Footer */}
        <footer className="app-footer">
          <div className="app-footer-container">
            <span className="app-footer-brand">
              IdeaGen — Evidence-Grounded Research Discovery & Opportunity Evolution Archive
            </span>
            <span className="app-footer-tagline">
              Read-only explorer · Proof-of-concept research prototype
            </span>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

export default App;
