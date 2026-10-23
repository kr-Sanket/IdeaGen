import React, { useState, useEffect } from 'react';
import { Search, BookOpen } from 'lucide-react';
import { SearchPapersView } from './SearchPapersView';
import { CorpusInspectorView } from './CorpusInspectorView';
import type { ResearchProject } from '../../types';

interface ProjectPapersViewProps {
  project: ResearchProject;
  initialTab?: 'search' | 'corpus';
  initialPaperId?: string | null;
  onNavigate: (route: string) => void;
  onInspectEvidence: (statementId: string) => void;
}

export const ProjectPapersView: React.FC<ProjectPapersViewProps> = ({
  project,
  initialTab = 'search',
  initialPaperId = null,
  onNavigate,
  onInspectEvidence,
}) => {
  const [activeTab, setActiveTab] = useState<'search' | 'corpus'>(initialTab);

  useEffect(() => {
    if (initialTab) {
      setActiveTab(initialTab);
    }
  }, [initialTab]);

  return (
    <div className="view-container animate-fade-in">
      {/* Sub-view Switcher Tabs */}
      <div className="subview-nav-bar">
        <div className="subview-nav-tabs">
          <button
            type="button"
            className={`subview-nav-tab ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
            id="tab-search-ingest"
          >
            <Search size={15} />
            <span>Search &amp; Ingest Literature</span>
          </button>

          <button
            type="button"
            className={`subview-nav-tab ${activeTab === 'corpus' ? 'active' : ''}`}
            onClick={() => setActiveTab('corpus')}
            id="tab-corpus-explorer"
          >
            <BookOpen size={15} />
            <span>Corpus Explorer &amp; Extracted Passages</span>
          </button>
        </div>

        <div className="subview-context-note">
          <span className="text-tertiary" style={{ fontSize: '0.75rem' }}>
            Project: <strong className="text-secondary">{project.name}</strong> · Cutoff: {project.temporalCutoff || 'None'}
          </span>
        </div>
      </div>

      {/* Render Active Subview */}
      {activeTab === 'search' ? (
        <SearchPapersView onNavigate={onNavigate} project={project} />
      ) : (
        <CorpusInspectorView
          project={project}
          initialPaperId={initialPaperId}
          onInspectEvidence={onInspectEvidence}
          onNavigate={onNavigate}
        />
      )}
    </div>
  );
};
