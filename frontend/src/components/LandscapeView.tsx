import React, { useEffect, useState } from 'react';
import type { StatementResponse } from '../types';
import { api } from '../api/client';
import { 
  GitFork, 
  HelpCircle, 
  Wrench, 
  CheckCircle2, 
  AlertTriangle, 
  Sparkles, 
  Bookmark, 
  Loader2, 
  AlertCircle
} from 'lucide-react';

interface LandscapeViewProps {
  onInspectEvidence: (statementId: string) => void;
}

export const LandscapeView: React.FC<LandscapeViewProps> = ({ onInspectEvidence }) => {
  const [statements, setStatements] = useState<StatementResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<string>('ALL');

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getStatements()
      .then((data) => setStatements(data))
      .catch((err) => setError(err.message || 'Failed to load research landscape statements'))
      .finally(() => setLoading(false));
  }, []);

  const filteredStatements = filterType === 'ALL'
    ? statements
    : statements.filter(s => s.statement_type?.toUpperCase() === filterType);

  // Categorized counts
  const problemCount = statements.filter(s => s.statement_type?.toUpperCase() === 'PROBLEM').length;
  const methodCount = statements.filter(s => s.statement_type?.toUpperCase() === 'METHOD').length;
  const findingCount = statements.filter(s => s.statement_type?.toUpperCase() === 'FINDING').length;
  const limitationCount = statements.filter(s => s.statement_type?.toUpperCase() === 'LIMITATION').length;
  const futureWorkCount = statements.filter(s => s.statement_type?.toUpperCase() === 'FUTURE_WORK').length;

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <GitFork className="w-6 h-6 text-cyan-400" />
            <span>Research Landscape</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Categorized knowledge extracted from literature (Problems, Methods, Findings, Limitations, Future Work).
          </p>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Failed to fetch landscape data</p>
            <p className="text-xs text-rose-300/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin text-cyan-400 mb-3" />
          <p className="text-sm">Synthesizing research landscape from corpus statements...</p>
        </div>
      )}

      {/* Category Pills Bar */}
      {!loading && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <button
            onClick={() => setFilterType('PROBLEM')}
            className={`p-3.5 rounded-xl border text-left transition ${
              filterType === 'PROBLEM'
                ? 'bg-rose-950/50 border-rose-600 text-rose-200'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center space-x-2 text-xs font-mono uppercase">
              <HelpCircle className="w-3.5 h-3.5 text-rose-400" />
              <span>Problems</span>
            </div>
            <span className="text-xl font-bold text-white block mt-1">{problemCount}</span>
          </button>

          <button
            onClick={() => setFilterType('METHOD')}
            className={`p-3.5 rounded-xl border text-left transition ${
              filterType === 'METHOD'
                ? 'bg-blue-950/50 border-blue-600 text-blue-200'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center space-x-2 text-xs font-mono uppercase">
              <Wrench className="w-3.5 h-3.5 text-blue-400" />
              <span>Methods</span>
            </div>
            <span className="text-xl font-bold text-white block mt-1">{methodCount}</span>
          </button>

          <button
            onClick={() => setFilterType('FINDING')}
            className={`p-3.5 rounded-xl border text-left transition ${
              filterType === 'FINDING'
                ? 'bg-emerald-950/50 border-emerald-600 text-emerald-200'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center space-x-2 text-xs font-mono uppercase">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              <span>Findings</span>
            </div>
            <span className="text-xl font-bold text-white block mt-1">{findingCount}</span>
          </button>

          <button
            onClick={() => setFilterType('LIMITATION')}
            className={`p-3.5 rounded-xl border text-left transition ${
              filterType === 'LIMITATION'
                ? 'bg-amber-950/50 border-amber-600 text-amber-200'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center space-x-2 text-xs font-mono uppercase">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>Limitations</span>
            </div>
            <span className="text-xl font-bold text-white block mt-1">{limitationCount}</span>
          </button>

          <button
            onClick={() => setFilterType('FUTURE_WORK')}
            className={`p-3.5 rounded-xl border text-left transition ${
              filterType === 'FUTURE_WORK'
                ? 'bg-purple-950/50 border-purple-600 text-purple-200'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center space-x-2 text-xs font-mono uppercase">
              <Sparkles className="w-3.5 h-3.5 text-purple-400" />
              <span>Future Work</span>
            </div>
            <span className="text-xl font-bold text-white block mt-1">{futureWorkCount}</span>
          </button>
        </div>
      )}

      {/* Filter Selector Header */}
      {!loading && (
        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-mono">Filter View:</span>
            <button
              onClick={() => setFilterType('ALL')}
              className={`px-2.5 py-1 rounded text-xs font-mono ${
                filterType === 'ALL'
                  ? 'bg-cyan-600 text-white font-semibold'
                  : 'bg-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              SHOW ALL ({statements.length})
            </button>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!loading && filteredStatements.length === 0 && (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800 text-slate-400 space-y-2">
          <Bookmark className="w-8 h-8 text-slate-500 mx-auto" />
          <h3 className="text-sm font-semibold text-slate-200">No Research Statements Found</h3>
          <p className="text-xs max-w-sm mx-auto text-slate-500">
            No statements match the selected filter criteria in the extracted literature database.
          </p>
        </div>
      )}

      {/* Statements List */}
      <div className="space-y-3">
        {filteredStatements.map((stmt) => (
          <div
            key={stmt.id}
            className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition flex items-start justify-between gap-4 group"
          >
            <div className="space-y-1.5 flex-1">
              <div className="flex items-center space-x-2">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase bg-slate-800 text-cyan-300 border border-slate-700">
                  {stmt.statement_type}
                </span>
                {stmt.section && (
                  <span className="text-[11px] text-slate-500 font-mono">
                    § {stmt.section}
                  </span>
                )}
                {stmt.paper_title && (
                  <span className="text-[11px] text-slate-400 font-sans truncate max-w-md">
                    — {stmt.paper_title}
                  </span>
                )}
              </div>
              <p className="text-sm text-slate-200 leading-relaxed">
                "{stmt.statement_text}"
              </p>
            </div>

            <button
              onClick={() => onInspectEvidence(stmt.id)}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-400 text-xs font-mono font-medium transition shrink-0 group-hover:bg-cyan-950 group-hover:text-cyan-300 border border-transparent group-hover:border-cyan-800/60"
            >
              Inspect Evidence →
            </button>
          </div>
        ))}
      </div>

    </div>
  );
};
