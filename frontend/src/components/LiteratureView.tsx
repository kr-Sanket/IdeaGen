import React, { useEffect, useState } from 'react';
import type { PaperResponse, ResearchKnowledgeResponse } from '../types';
import { api } from '../api/client';
import { 
  BookOpen, 
  FileText, 
  Calendar, 
  User, 
  Layers, 
  CheckCircle2, 
  X, 
  Loader2, 
  AlertCircle,
  ExternalLink,
  Tag,
  ChevronRight
} from 'lucide-react';

interface LiteratureViewProps {
  onInspectEvidence: (statementId: string) => void;
}

export const LiteratureView: React.FC<LiteratureViewProps> = ({ onInspectEvidence }) => {
  const [papers, setPapers] = useState<PaperResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Modal / Selected Paper Detail
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);
  const [paperDetail, setPaperDetail] = useState<PaperResponse | null>(null);
  const [knowledge, setKnowledge] = useState<ResearchKnowledgeResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api.getPapers()
      .then((data) => setPapers(data))
      .catch((err) => setError(err.message || 'Failed to load literature'))
      .finally(() => setLoading(false));
  }, []);

  const openPaperDetail = (paperId: string) => {
    setSelectedPaperId(paperId);
    setDetailLoading(true);
    setDetailError(null);
    setPaperDetail(null);
    setKnowledge(null);

    Promise.all([
      api.getPaperDetail(paperId),
      api.getPaperKnowledge(paperId).catch(() => null) // Knowledge might be 404 if not extracted yet
    ])
      .then(([p, k]) => {
        setPaperDetail(p);
        setKnowledge(k);
      })
      .catch((err) => setDetailError(err.message || 'Failed to load paper details'))
      .finally(() => setDetailLoading(false));
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <BookOpen className="w-6 h-6 text-indigo-400" />
            <span>Literature Repository</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Ingested corpus documents, passage structures, and extracted research knowledge.
          </p>
        </div>
        <div className="text-xs font-mono px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400">
          Total Papers: <span className="text-white font-bold">{papers.length}</span>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-300 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Failed to fetch literature</p>
            <p className="text-xs text-rose-300/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center py-20 text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-400 mb-3" />
          <p className="text-sm">Loading ingested paper index...</p>
        </div>
      )}

      {!loading && papers.length === 0 && !error && (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800 space-y-3">
          <FileText className="w-10 h-10 text-slate-500 mx-auto" />
          <h3 className="text-base font-semibold text-slate-200">No Ingested Papers Found</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            No research papers are available in the backend database. Ingest PDF papers via the pipeline script to view them here.
          </p>
        </div>
      )}

      {/* Papers Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {papers.map((paper) => (
          <div
            key={paper.id}
            onClick={() => openPaperDetail(paper.id)}
            className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 transition cursor-pointer flex flex-col justify-between group space-y-4"
          >
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-800 text-slate-400 border border-slate-700">
                  {paper.source_type}
                </span>
                {paper.has_knowledge ? (
                  <span className="flex items-center space-x-1 text-[11px] font-medium text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Knowledge Extracted</span>
                  </span>
                ) : (
                  <span className="text-[11px] text-slate-500 font-mono">Pending Extraction</span>
                )}
              </div>

              <h3 className="text-base font-semibold text-slate-100 group-hover:text-indigo-300 transition line-clamp-2">
                {paper.title || paper.source_id || 'Untitled Paper'}
              </h3>

              <p className="text-xs text-slate-400 line-clamp-3 leading-relaxed">
                {paper.abstract || 'No abstract text available.'}
              </p>
            </div>

            {/* Metadata Footer */}
            <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
              <div className="flex items-center space-x-3">
                <span className="flex items-center space-x-1">
                  <Layers className="w-3.5 h-3.5 text-slate-500" />
                  <span>{paper.passages_count} passages</span>
                </span>
                <span className="flex items-center space-x-1">
                  <Tag className="w-3.5 h-3.5 text-slate-500" />
                  <span>{paper.statements_count} statements</span>
                </span>
              </div>
              <span className="text-indigo-400 font-medium flex items-center space-x-1 group-hover:translate-x-0.5 transition-transform">
                <span>Details</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </span>
            </div>

          </div>
        ))}
      </div>

      {/* Selected Paper Details Modal */}
      {selectedPaperId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="w-full max-w-4xl max-h-[90vh] bg-slate-900 border border-slate-700/80 rounded-2xl flex flex-col shadow-2xl overflow-hidden">
            
            {/* Modal Header */}
            <div className="p-5 border-b border-slate-800 bg-slate-950/60 flex items-center justify-between">
              <div className="flex items-center space-x-2 text-xs font-mono text-indigo-400 uppercase tracking-wider">
                <FileText className="w-4 h-4" />
                <span>Paper Knowledge Overview</span>
              </div>
              <button
                onClick={() => setSelectedPaperId(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {detailLoading && (
                <div className="flex flex-col items-center justify-center py-20 text-slate-400">
                  <Loader2 className="w-8 h-8 animate-spin text-indigo-400 mb-3" />
                  <p className="text-sm">Fetching paper details & knowledge schema...</p>
                </div>
              )}

              {detailError && (
                <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm">
                  {detailError}
                </div>
              )}

              {paperDetail && !detailLoading && (
                <>
                  {/* Paper Meta Card */}
                  <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                    <h2 className="text-xl font-bold text-white leading-snug">
                      {paperDetail.title || paperDetail.source_id}
                    </h2>
                    
                    <div className="flex flex-wrap gap-4 text-xs text-slate-400">
                      {paperDetail.authors && paperDetail.authors.length > 0 && (
                        <span className="flex items-center space-x-1">
                          <User className="w-3.5 h-3.5 text-slate-500" />
                          <span>{paperDetail.authors.join(', ')}</span>
                        </span>
                      )}
                      {paperDetail.publication_date && (
                        <span className="flex items-center space-x-1">
                          <Calendar className="w-3.5 h-3.5 text-slate-500" />
                          <span>{paperDetail.publication_date}</span>
                        </span>
                      )}
                      {paperDetail.doi && (
                        <span className="flex items-center space-x-1 font-mono text-indigo-400">
                          <ExternalLink className="w-3.5 h-3.5" />
                          <span>DOI: {paperDetail.doi}</span>
                        </span>
                      )}
                    </div>

                    <div className="text-xs text-slate-300 pt-2 border-t border-slate-800/80 leading-relaxed">
                      <span className="font-semibold text-slate-400 block mb-1">Abstract</span>
                      {paperDetail.abstract || 'No abstract text stored.'}
                    </div>
                  </div>

                  {/* Sections list */}
                  {paperDetail.sections && paperDetail.sections.length > 0 && (
                    <div className="space-y-2">
                      <span className="text-xs font-mono uppercase text-slate-400 block">Extracted Sections</span>
                      <div className="flex flex-wrap gap-1.5">
                        {paperDetail.sections.map((sec, idx) => (
                          <span key={idx} className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300 text-xs font-mono">
                            {sec}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Research Knowledge */}
                  {knowledge ? (
                    <div className="space-y-4 pt-4 border-t border-slate-800">
                      <h3 className="text-sm font-bold text-indigo-300 uppercase font-mono tracking-wider">
                        Extracted Research Knowledge
                      </h3>

                      {knowledge.research_problem && (
                        <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-1">
                          <span className="text-xs font-mono text-indigo-400 uppercase">Research Problem</span>
                          <p className="text-sm text-slate-200">{knowledge.research_problem}</p>
                        </div>
                      )}

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {knowledge.findings && knowledge.findings.length > 0 && (
                          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                            <span className="text-xs font-mono text-emerald-400 uppercase block">Key Findings</span>
                            <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                              {knowledge.findings.map((f, i) => (
                                <li key={i}>{f}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {knowledge.limitations && knowledge.limitations.length > 0 && (
                          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-2">
                            <span className="text-xs font-mono text-amber-400 uppercase block">Limitations</span>
                            <ul className="list-disc list-inside text-xs text-slate-300 space-y-1">
                              {knowledge.limitations.map((l, i) => (
                                <li key={i}>{l}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Research Statements */}
                      {knowledge.statements && knowledge.statements.length > 0 && (
                        <div className="space-y-3 pt-2">
                          <span className="text-xs font-mono uppercase text-slate-400 block">
                            Extracted Statements ({knowledge.statements.length})
                          </span>
                          <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                            {knowledge.statements.map((stmt) => (
                              <div
                                key={stmt.id}
                                onClick={() => onInspectEvidence(stmt.id)}
                                className="p-3 rounded-lg bg-slate-950 border border-slate-800 hover:border-indigo-600/60 cursor-pointer transition flex items-center justify-between group"
                              >
                                <div className="space-y-0.5 max-w-xl">
                                  <span className="px-2 py-0.2 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                                    {stmt.statement_type}
                                  </span>
                                  <p className="text-xs text-slate-200 group-hover:text-indigo-200 line-clamp-2">
                                    "{stmt.statement_text}"
                                  </p>
                                </div>
                                <span className="text-[11px] text-indigo-400 font-mono shrink-0 ml-2">
                                  Trace Evidence →
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                    </div>
                  ) : (
                    <div className="p-6 text-center rounded-xl bg-slate-950 border border-slate-800 text-slate-400 text-xs">
                      No research knowledge extracted for this paper yet.
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-slate-800 bg-slate-950/60 flex justify-end">
              <button
                onClick={() => setSelectedPaperId(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition"
              >
                Close Details
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
};
