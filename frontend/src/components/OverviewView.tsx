import React, { useEffect, useState } from 'react';
import type { WorkspaceStats } from '../types';
import { api } from '../api/client';
import type { TabView } from './Navbar';
import { 
  BookOpen, 
  FileSearch, 
  Sparkles, 
  Lightbulb, 
  ArrowRight, 
  RefreshCw, 
  AlertCircle,
  Database,
  Layers
} from 'lucide-react';

interface OverviewViewProps {
  onSelectTab: (tab: TabView) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ onSelectTab }) => {
  const [stats, setStats] = useState<WorkspaceStats | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = () => {
    setLoading(true);
    setError(null);
    api.getStats()
      .then((data) => setStats(data))
      .catch((err) => setError(err.message || 'Failed to load workspace stats'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchStats();
  }, []);

  const hasData = stats && (stats.total_papers > 0 || stats.total_statements > 0 || stats.total_gaps > 0);

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Research Workspace Dashboard</h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time pipeline metrics synthesized from literature ingestion and gap validation.
          </p>
        </div>
        <button
          onClick={fetchStats}
          disabled={loading}
          className="flex items-center space-x-2 px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-indigo-400' : ''}`} />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Backend Connection Warning</p>
            <p className="text-xs text-rose-300/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {loading && !stats && (
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-28 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse" />
          ))}
        </div>
      )}

      {/* Metrics Grid */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          
          {/* Card 1: Total Papers */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition relative overflow-hidden group">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Ingested Papers</span>
              <div className="w-8 h-8 rounded-lg bg-blue-950 text-blue-400 flex items-center justify-center border border-blue-800/50">
                <BookOpen className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <span className="text-3xl font-bold text-white tracking-tight">{stats.total_papers}</span>
              <span className="text-xs text-slate-500 block mt-1">
                {stats.processed_papers} fully analyzed
              </span>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 flex justify-end">
              <button
                onClick={() => onSelectTab('literature')}
                className="text-xs text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform"
              >
                <span>View Literature</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Card 2: Research Knowledge */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition relative overflow-hidden group">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Research Statements</span>
              <div className="w-8 h-8 rounded-lg bg-cyan-950 text-cyan-400 flex items-center justify-center border border-cyan-800/50">
                <FileSearch className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <span className="text-3xl font-bold text-white tracking-tight">{stats.total_statements}</span>
              <span className="text-xs text-slate-500 block mt-1">
                Extracted ground claims & facts
              </span>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 flex justify-end">
              <button
                onClick={() => onSelectTab('landscape')}
                className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform"
              >
                <span>Inspect Landscape</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Card 3: Research Gaps */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition relative overflow-hidden group">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Research Gaps</span>
              <div className="w-8 h-8 rounded-lg bg-purple-950 text-purple-400 flex items-center justify-center border border-purple-800/50">
                <Sparkles className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3 flex items-baseline space-x-2">
              <span className="text-3xl font-bold text-white tracking-tight">{stats.total_gaps}</span>
              <span className="text-xs text-emerald-400 font-semibold px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800/60">
                {stats.validated_gaps} Validated
              </span>
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {stats.candidate_gaps} Candidate · {stats.rejected_gaps} Rejected
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 flex justify-end">
              <button
                onClick={() => onSelectTab('gaps')}
                className="text-xs text-purple-400 hover:text-purple-300 flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform"
              >
                <span>Explore Gaps</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Card 4: Opportunities */}
          <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition relative overflow-hidden group">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Opportunities</span>
              <div className="w-8 h-8 rounded-lg bg-amber-950 text-amber-400 flex items-center justify-center border border-amber-800/50">
                <Lightbulb className="w-4 h-4" />
              </div>
            </div>
            <div className="mt-3">
              <span className="text-3xl font-bold text-white tracking-tight">{stats.total_opportunities}</span>
              <span className="text-xs text-slate-500 block mt-1">
                Evidence-grounded proposals
              </span>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/80 flex justify-end">
              <button
                onClick={() => onSelectTab('opportunities')}
                className="text-xs text-amber-400 hover:text-amber-300 flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform"
              >
                <span>Open Studio</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>

        </div>
      )}

      {/* Workflow Navigation Cards */}
      <div className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800 space-y-4">
        <h2 className="text-base font-semibold text-slate-200 flex items-center space-x-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span>IdeaGen Pipeline Workflow</span>
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          <div 
            onClick={() => onSelectTab('literature')}
            className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-indigo-600/50 cursor-pointer transition group"
          >
            <div className="text-xs font-mono text-slate-500 font-semibold mb-1">01. INGESTION</div>
            <h4 className="text-sm font-medium text-slate-200 group-hover:text-indigo-300">Literature</h4>
            <p className="text-xs text-slate-400 mt-1">Ingest papers & parse passages</p>
          </div>

          <div 
            onClick={() => onSelectTab('landscape')}
            className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-cyan-600/50 cursor-pointer transition group"
          >
            <div className="text-xs font-mono text-slate-500 font-semibold mb-1">02. KNOWLEDGE</div>
            <h4 className="text-sm font-medium text-slate-200 group-hover:text-cyan-300">Landscape</h4>
            <p className="text-xs text-slate-400 mt-1">Extract facts, claims & methods</p>
          </div>

          <div 
            onClick={() => onSelectTab('gaps')}
            className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-purple-600/50 cursor-pointer transition group"
          >
            <div className="text-xs font-mono text-slate-500 font-semibold mb-1">03. DISCOVERY</div>
            <h4 className="text-sm font-medium text-slate-200 group-hover:text-purple-300">Research Gaps</h4>
            <p className="text-xs text-slate-400 mt-1">Discover & validate gaps</p>
          </div>

          <div 
            onClick={() => onSelectTab('opportunities')}
            className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 hover:border-amber-600/50 cursor-pointer transition group"
          >
            <div className="text-xs font-mono text-slate-500 font-semibold mb-1">04. PROPOSALS</div>
            <h4 className="text-sm font-medium text-slate-200 group-hover:text-amber-300">Opportunity Studio</h4>
            <p className="text-xs text-slate-400 mt-1">Synthesize grounded proposals</p>
          </div>

          <div 
            onClick={() => onSelectTab('evaluation')}
            className="p-4 rounded-xl bg-slate-950/40 border border-slate-850 hover:border-slate-700 cursor-pointer transition group"
          >
            <div className="text-xs font-mono text-slate-500 font-semibold mb-1">05. EVALUATION</div>
            <h4 className="text-sm font-medium text-slate-400 group-hover:text-slate-300">Validation</h4>
            <p className="text-xs text-slate-500 mt-1">Phase 7 experimental bench</p>
          </div>
        </div>
      </div>

      {/* Empty State when no papers/gaps present */}
      {!loading && stats && !hasData && (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800/80 space-y-4">
          <div className="w-12 h-12 rounded-full bg-slate-800 text-slate-400 flex items-center justify-center mx-auto">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-200">No Research Data Ingested Yet</h3>
            <p className="text-sm text-slate-400 max-w-md mx-auto mt-1">
              Your research database is currently empty. Run the backend ingestion pipeline or check backend connections to populate literature data.
            </p>
          </div>
          <button
            onClick={() => onSelectTab('literature')}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 transition"
          >
            <span>Go to Literature Tab</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}

    </div>
  );
};
