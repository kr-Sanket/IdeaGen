import React, { useEffect, useState } from 'react';
import { X, FileText, Bookmark, Layers, AlertCircle, Loader2 } from 'lucide-react';
import type { EvidenceDetail } from '../types';
import { api } from '../api/client';

interface EvidenceDrawerProps {
  statementId: string | null;
  onClose: () => void;
}

export const EvidenceDrawer: React.FC<EvidenceDrawerProps> = ({ statementId, onClose }) => {
  const [evidence, setEvidence] = useState<EvidenceDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!statementId) {
      setEvidence(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.getEvidenceDetail(statementId)
      .then((data) => setEvidence(data))
      .catch((err) => setError(err.message || 'Failed to load evidence detail'))
      .finally(() => setLoading(false));
  }, [statementId]);

  if (!statementId) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="w-full max-w-xl bg-slate-900 border-l border-slate-700/60 text-slate-100 flex flex-col h-full shadow-2xl animate-in slide-in-from-right duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-800 bg-slate-950/50">
          <div className="flex items-center space-x-2">
            <Bookmark className="w-5 h-5 text-indigo-400" />
            <h3 className="font-semibold text-lg text-slate-100">Evidence Provenance</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {loading && (
            <div className="flex flex-col items-center justify-center py-20 text-slate-400">
              <Loader2 className="w-8 h-8 animate-spin text-indigo-400 mb-3" />
              <p className="text-sm">Fetching ground evidence provenance...</p>
            </div>
          )}

          {error && (
            <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-sm flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Unable to fetch evidence detail</p>
                <p className="text-xs text-rose-300/80 mt-1">{error}</p>
              </div>
            </div>
          )}

          {evidence && !loading && (
            <>
              {/* Paper Source */}
              <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/50 space-y-2">
                <div className="flex items-center space-x-2 text-xs text-indigo-400 font-mono uppercase tracking-wider">
                  <FileText className="w-4 h-4" />
                  <span>Source Paper</span>
                </div>
                <h4 className="font-medium text-slate-200 text-base">
                  {evidence.paper_title || `Paper ID: ${evidence.paper_id}`}
                </h4>
                <div className="flex flex-wrap gap-2 pt-1 text-xs text-slate-400">
                  <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                    ID: {evidence.paper_id.slice(0, 8)}...
                  </span>
                </div>
              </div>

              {/* Location Provenance */}
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3.5 rounded-xl bg-slate-800/30 border border-slate-700/40">
                  <span className="text-xs text-slate-400 block mb-1">Section</span>
                  <span className="font-medium text-slate-200 text-sm">
                    {evidence.section || 'Unspecified section'}
                  </span>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-800/30 border border-slate-700/40">
                  <span className="text-xs text-slate-400 block mb-1">Page Range</span>
                  <span className="font-medium text-slate-200 text-sm">
                    {evidence.page_start !== null && evidence.page_start !== undefined
                      ? `Page ${evidence.page_start}${evidence.page_end && evidence.page_end !== evidence.page_start ? `–${evidence.page_end}` : ''}`
                      : 'N/A'}
                  </span>
                </div>
              </div>

              {/* Extracted Research Statement */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-mono uppercase text-slate-400">Extracted Statement</span>
                  <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-indigo-950/70 text-indigo-300 border border-indigo-800/60">
                    {evidence.statement_type}
                  </span>
                </div>
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 text-sm italic leading-relaxed">
                  "{evidence.statement_text}"
                </div>
              </div>

              {/* Passage Text */}
              {evidence.passage_text && (
                <div className="space-y-2">
                  <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono uppercase">
                    <Layers className="w-4 h-4 text-emerald-400" />
                    <span>Exact Source Passage</span>
                    {evidence.passage_index !== null && (
                      <span className="text-slate-500">(Passage #{evidence.passage_index})</span>
                    )}
                  </div>
                  <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/80 text-slate-300 text-xs font-mono leading-relaxed max-h-60 overflow-y-auto">
                    {evidence.passage_text}
                  </div>
                </div>
              )}

              {/* Confidence Indicator */}
              {evidence.confidence !== null && evidence.confidence !== undefined && (
                <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                  <span>Extraction Confidence</span>
                  <span className="font-mono text-emerald-400 font-medium">
                    {(evidence.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-950/50 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm bg-slate-800 hover:bg-slate-700 text-slate-200 transition"
          >
            Close Panel
          </button>
        </div>
      </div>
    </div>
  );
};
