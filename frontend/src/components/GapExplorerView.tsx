import React, { useEffect, useState } from 'react';
import type { GapResponse } from '../types';
import { api } from '../api/client';
import type { TabView } from './Navbar';
import { 
  Sparkles, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  FileText, 
  Layers, 
  Loader2, 
  AlertCircle,
  Lightbulb,
  Play
} from 'lucide-react';

interface GapExplorerViewProps {
  onInspectEvidence: (statementId: string) => void;
  onSelectTab: (tab: TabView) => void;
  onGenerateOpportunity: (gapId: string) => void;
}

export const GapExplorerView: React.FC<GapExplorerViewProps> = ({
  onGenerateOpportunity,
}) => {
  const [gaps, setGaps] = useState<GapResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [discovering, setDiscovering] = useState<boolean>(false);
  const [generatingGapId, setGeneratingGapId] = useState<string | null>(null);

  const [filterStatus, setFilterStatus] = useState<string>('ALL');

  const fetchGaps = () => {
    setLoading(true);
    setError(null);
    api.getGaps()
      .then((data) => setGaps(data))
      .catch((err) => setError(err.message || 'Failed to load research gaps'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchGaps();
  }, []);

  const handleTriggerDiscovery = async () => {
    setDiscovering(true);
    setError(null);
    try {
      await api.discoverGaps();
      fetchGaps();
    } catch (err: any) {
      setError(err.message || 'Failed to trigger gap discovery');
    } finally {
      setDiscovering(false);
    }
  };

  const handleGenerate = async (gapId: string) => {
    setGeneratingGapId(gapId);
    try {
      await api.generateOpportunity(gapId);
      onGenerateOpportunity(gapId);
    } catch (err: any) {
      setError(err.message || 'Failed to generate research opportunity');
      setGeneratingGapId(null);
    }
  };

  const filteredGaps = filterStatus === 'ALL'
    ? gaps
    : gaps.filter((g) => g.status === filterStatus);

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <Sparkles className="w-6 h-6 text-purple-400" />
            <span>Research Gap Explorer</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Grounded gap candidates discovered across literature and validated against evidence.
          </p>
        </div>

        <button
          onClick={handleTriggerDiscovery}
          disabled={discovering}
          className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold shadow-lg shadow-purple-600/20 transition disabled:opacity-50"
        >
          {discovering ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing Gaps...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              <span>Run Gap Discovery</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Gap Pipeline Notification</p>
            <p className="text-xs text-rose-300/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Filter status tabs */}
      {!loading && (
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <button
            onClick={() => setFilterStatus('ALL')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition ${
              filterStatus === 'ALL'
                ? 'bg-slate-800 text-white font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            ALL ({gaps.length})
          </button>
          <button
            onClick={() => setFilterStatus('VALIDATED')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition flex items-center space-x-1 ${
              filterStatus === 'VALIDATED'
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-800 font-semibold'
                : 'text-emerald-400/70 hover:text-emerald-300'
            }`}
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>VALIDATED ({gaps.filter((g) => g.status === 'VALIDATED').length})</span>
          </button>
          <button
            onClick={() => setFilterStatus('CANDIDATE')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition flex items-center space-x-1 ${
              filterStatus === 'CANDIDATE'
                ? 'bg-amber-950 text-amber-300 border border-amber-800 font-semibold'
                : 'text-amber-400/70 hover:text-amber-300'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>CANDIDATE ({gaps.filter((g) => g.status === 'CANDIDATE').length})</span>
          </button>
          <button
            onClick={() => setFilterStatus('REJECTED')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition flex items-center space-x-1 ${
              filterStatus === 'REJECTED'
                ? 'bg-rose-950 text-rose-300 border border-rose-800 font-semibold'
                : 'text-rose-400/70 hover:text-rose-300'
            }`}
          >
            <XCircle className="w-3.5 h-3.5" />
            <span>REJECTED ({gaps.filter((g) => g.status === 'REJECTED').length})</span>
          </button>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin text-purple-400 mb-3" />
          <p className="text-sm">Fetching discovered research gaps...</p>
        </div>
      )}

      {!loading && filteredGaps.length === 0 && (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800 text-slate-400 space-y-3">
          <Sparkles className="w-8 h-8 text-slate-500 mx-auto" />
          <h3 className="text-sm font-semibold text-slate-200">No Research Gaps Available</h3>
          <p className="text-xs max-w-sm mx-auto text-slate-500">
            Click "Run Gap Discovery" above to trigger Phase 4 gap synthesis over ingested literature.
          </p>
        </div>
      )}

      {/* Gaps List */}
      <div className="space-y-4">
        {filteredGaps.map((gap) => {
          const isGenerating = generatingGapId === gap.id;

          return (
            <div
              key={gap.id}
              className={`p-6 rounded-2xl bg-slate-900/80 border transition space-y-4 ${
                gap.status === 'VALIDATED'
                  ? 'border-emerald-900/60 hover:border-emerald-700/80'
                  : gap.status === 'REJECTED'
                  ? 'border-rose-900/40 opacity-75'
                  : 'border-slate-800 hover:border-slate-700'
              }`}
            >
              {/* Gap Header / Status */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {gap.status === 'VALIDATED' && (
                    <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center space-x-1">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>VALIDATED</span>
                    </span>
                  )}
                  {gap.status === 'CANDIDATE' && (
                    <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-amber-950 text-amber-400 border border-amber-800 flex items-center space-x-1">
                      <Clock className="w-3.5 h-3.5" />
                      <span>CANDIDATE</span>
                    </span>
                  )}
                  {gap.status === 'REJECTED' && (
                    <span className="px-2.5 py-1 rounded-md text-xs font-mono font-bold bg-rose-950 text-rose-400 border border-rose-800 flex items-center space-x-1">
                      <XCircle className="w-3.5 h-3.5" />
                      <span>REJECTED</span>
                    </span>
                  )}

                  {gap.validation_confidence !== null && gap.validation_confidence !== undefined && (
                    <span className="text-xs font-mono text-slate-400">
                      Confidence: {(gap.validation_confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>

                <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
                  <FileText className="w-3.5 h-3.5 text-slate-500" />
                  <span>{gap.evidence_count} evidence items</span>
                </div>
              </div>

              {/* Gap Statement */}
              <div className="space-y-1">
                <span className="text-xs font-mono text-purple-400 uppercase tracking-wider block">
                  Identified Gap Statement
                </span>
                <h3 className="text-base font-bold text-white leading-snug">
                  {gap.gap_statement}
                </h3>
              </div>

              {/* Research Problem context */}
              {gap.research_problem && (
                <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-300">
                  <span className="font-mono text-slate-400 block mb-1">Target Research Problem:</span>
                  {gap.research_problem}
                </div>
              )}

              {/* Validation Reasoning */}
              {gap.validation_reasoning && (
                <div className="text-xs text-slate-400 space-y-1 pt-1">
                  <span className="font-mono text-slate-500 block">Validation Analysis:</span>
                  <p className="italic text-slate-300 leading-relaxed bg-slate-950/40 p-3 rounded-lg border border-slate-800/60">
                    "{gap.validation_reasoning}"
                  </p>
                </div>
              )}

              {/* Supporting papers / Actions */}
              <div className="pt-3 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="text-xs text-slate-400 flex items-center space-x-2">
                  <Layers className="w-3.5 h-3.5 text-slate-500" />
                  <span>
                    Supporting Papers: {gap.supporting_papers ? gap.supporting_papers.length : 0}
                  </span>
                </div>

                {gap.status === 'VALIDATED' && (
                  <button
                    onClick={() => handleGenerate(gap.id)}
                    disabled={isGenerating}
                    className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold shadow-lg shadow-amber-600/20 transition disabled:opacity-50"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Synthesizing Opportunity...</span>
                      </>
                    ) : (
                      <>
                        <Lightbulb className="w-4 h-4" />
                        <span>Synthesize Opportunity Proposal →</span>
                      </>
                    )}
                  </button>
                )}
              </div>

            </div>
          );
        })}
      </div>

    </div>
  );
};
