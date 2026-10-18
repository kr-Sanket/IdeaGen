import { useState } from 'react';
import { Navbar, type TabView } from './components/Navbar';
import { OverviewView } from './components/OverviewView';
import { LiteratureView } from './components/LiteratureView';
import { LandscapeView } from './components/LandscapeView';
import { GapExplorerView } from './components/GapExplorerView';
import { OpportunityStudioView } from './components/OpportunityStudioView';
import { EvaluationView } from './components/EvaluationView';
import { EvidenceDrawer } from './components/EvidenceDrawer';
import './App.css';

export function App() {
  const [currentTab, setCurrentTab] = useState<TabView>('overview');
  const [activeEvidenceStatementId, setActiveEvidenceStatementId] = useState<string | null>(null);

  const handleInspectEvidence = (statementId: string) => {
    setActiveEvidenceStatementId(statementId);
  };

  const handleCloseEvidence = () => {
    setActiveEvidenceStatementId(null);
  };

  const handleGenerateOpportunity = () => {
    setCurrentTab('opportunities');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased flex flex-col selection:bg-indigo-500 selection:text-white">
      
      {/* Top Navbar */}
      <Navbar currentTab={currentTab} onSelectTab={setCurrentTab} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        {currentTab === 'overview' && (
          <OverviewView onSelectTab={setCurrentTab} />
        )}

        {currentTab === 'literature' && (
          <LiteratureView onInspectEvidence={handleInspectEvidence} />
        )}

        {currentTab === 'landscape' && (
          <LandscapeView onInspectEvidence={handleInspectEvidence} />
        )}

        {currentTab === 'gaps' && (
          <GapExplorerView
            onInspectEvidence={handleInspectEvidence}
            onSelectTab={setCurrentTab}
            onGenerateOpportunity={handleGenerateOpportunity}
          />
        )}

        {currentTab === 'opportunities' && (
          <OpportunityStudioView
            onInspectEvidence={handleInspectEvidence}
          />
        )}

        {currentTab === 'evaluation' && (
          <EvaluationView />
        )}
      </main>

      {/* Reusable Ground Evidence Provenance Drawer */}
      <EvidenceDrawer
        statementId={activeEvidenceStatementId}
        onClose={handleCloseEvidence}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>IdeaGen — Evidence-Grounded Research Discovery System</span>
          <span className="font-mono text-slate-400">
            Phase 6 Pipeline Active · Potentially novel within analyzed corpus
          </span>
        </div>
      </footer>

    </div>
  );
}

export default App;
