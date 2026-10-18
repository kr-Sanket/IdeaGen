import React, { useEffect, useState } from 'react';
import type { OpportunityResponse, OpportunityReport } from '../types';
import { api } from '../api/client';
import { 
  Lightbulb, 
  ShieldCheck, 
  Loader2, 
  AlertCircle,
  Sparkles,
  Wrench,
  GitCompare,
  Gauge
} from 'lucide-react';

interface OpportunityStudioViewProps {
  onInspectEvidence: (statementId: string) => void;
  selectedOpportunityId?: string | null;
}

export const OpportunityStudioView: React.FC<OpportunityStudioViewProps> = ({
  onInspectEvidence,
  selectedOpportunityId,
}) => {
  const [opportunities, setOpportunities] = useState<OpportunityResponse[]>([]);
  const [activeOpportunityId, setActiveOpportunityId] = useState<string | null>(selectedOpportunityId || null);
  const [report, setReport] = useState<OpportunityReport | null>(null);
  
  const [loadingList, setLoadingList] = useState<boolean>(true);
  const [loadingReport, setLoadingReport] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoadingList(true);
    setError(null);
    api.getOpportunities()
      .then((data) => {
        setOpportunities(data);
        if (data.length > 0 && !activeOpportunityId) {
          setActiveOpportunityId(data[0].id);
        }
      })
      .catch((err) => setError(err.message || 'Failed to load research opportunities'))
      .finally(() => setLoadingList(false));
  }, []);

  useEffect(() => {
    if (!activeOpportunityId) {
      setReport(null);
      return;
    }
    setLoadingReport(true);
    setError(null);
    api.getOpportunityReport(activeOpportunityId)
      .then((rep) => setReport(rep))
      .catch((err) => setError(err.message || 'Failed to load opportunity report'))
      .finally(() => setLoadingReport(false));
  }, [activeOpportunityId]);

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <Lightbulb className="w-6 h-6 text-amber-400" />
            <span>Opportunity Studio</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Structured 16-section research proposals synthesized from validated literature gaps.
          </p>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800 text-rose-300 text-sm flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Opportunity Studio Error</p>
            <p className="text-xs text-rose-300/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {loadingList && (
        <div className="flex flex-col items-center justify-center py-20 text-slate-400">
          <Loader2 className="w-8 h-8 animate-spin text-amber-400 mb-3" />
          <p className="text-sm">Loading research opportunity index...</p>
        </div>
      )}

      {!loadingList && opportunities.length === 0 && !error && (
        <div className="p-12 text-center rounded-2xl bg-slate-900/40 border border-slate-800 text-slate-400 space-y-3">
          <Lightbulb className="w-10 h-10 text-slate-500 mx-auto" />
          <h3 className="text-base font-semibold text-slate-200">No Generated Opportunities</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Navigate to the Research Gaps tab and click "Synthesize Opportunity Proposal" on a validated gap to generate a proposal report here.
          </p>
        </div>
      )}

      {/* Main Layout: Selector Bar + Report View */}
      {opportunities.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Left Selector Sidebar */}
          <div className="space-y-3">
            <span className="text-xs font-mono uppercase text-slate-400 block tracking-wider">
              Generated Proposals ({opportunities.length})
            </span>
            <div className="space-y-2">
              {opportunities.map((opp) => (
                <div
                  key={opp.id}
                  onClick={() => setActiveOpportunityId(opp.id)}
                  className={`p-4 rounded-xl border cursor-pointer transition text-left space-y-2 ${
                    activeOpportunityId === opp.id
                      ? 'bg-amber-950/40 border-amber-500 text-white shadow-lg shadow-amber-500/10'
                      : 'bg-slate-900/70 border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-amber-400 font-semibold">
                      {(opp.confidence * 100).toFixed(0)}% CONFIDENCE
                    </span>
                    <span className="text-slate-500">
                      ID: {opp.id.slice(0, 6)}...
                    </span>
                  </div>
                  <h4 className="text-xs font-bold leading-snug line-clamp-2">
                    {opp.proposed_research_direction}
                  </h4>
                </div>
              ))}
            </div>
          </div>

          {/* Right Main Report View (Full 16 Sections) */}
          <div className="lg:col-span-3 space-y-6">
            {loadingReport && (
              <div className="flex flex-col items-center justify-center py-20 text-slate-400 rounded-2xl bg-slate-900/40 border border-slate-800">
                <Loader2 className="w-8 h-8 animate-spin text-amber-400 mb-3" />
                <p className="text-sm">Synthesizing 16-section opportunity report...</p>
              </div>
            )}

            {report && !loadingReport && (
              <div className="p-8 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-8 shadow-2xl">
                
                {/* 16. Cautious Novelty & Title Banner */}
                <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/60 flex items-start space-x-3 text-amber-200">
                  <ShieldCheck className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="text-xs font-mono uppercase font-bold tracking-wider text-amber-400 block mb-0.5">
                      Novelty Assessment Status
                    </span>
                    <p className="text-xs leading-relaxed font-mono">
                      "{report.cautious_novelty_statement}"
                    </p>
                  </div>
                </div>

                {/* Proposal Title Header */}
                <div className="space-y-2 border-b border-slate-800 pb-6">
                  <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
                    <span className="px-2.5 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                      VALIDATED OPPORTUNITY
                    </span>
                    <span>Confidence: {(report.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <h2 className="text-2xl font-extrabold text-white leading-tight">
                    {report.proposed_research_direction}
                  </h2>
                </div>

                {/* Section 1 & 4: Research Problem & Identified Gap */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <span className="text-xs font-mono uppercase text-indigo-400 block font-semibold">
                      1. Research Problem
                    </span>
                    <p className="text-sm text-slate-200 leading-relaxed">
                      {report.research_problem}
                    </p>
                  </div>

                  <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <span className="text-xs font-mono uppercase text-purple-400 block font-semibold">
                      4. Identified Gap
                    </span>
                    <p className="text-sm text-slate-200 leading-relaxed">
                      {report.identified_gap}
                    </p>
                  </div>
                </div>

                {/* Section 5: Why Existing Work Is Insufficient */}
                <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                  <span className="text-xs font-mono uppercase text-rose-400 block font-semibold">
                    5. Why Existing Work Is Insufficient
                  </span>
                  <p className="text-sm text-slate-300 leading-relaxed">
                    {report.why_existing_work_is_insufficient}
                  </p>
                </div>

                {/* Section 6 & 7: Proposed Research Direction & Reasoning */}
                <div className="p-6 rounded-xl bg-gradient-to-br from-slate-950 to-indigo-950/40 border border-indigo-900/60 space-y-4">
                  <div>
                    <span className="text-xs font-mono uppercase text-indigo-300 block font-semibold mb-1">
                      6. Proposed Research Direction
                    </span>
                    <h3 className="text-base font-bold text-white">
                      {report.proposed_research_direction}
                    </h3>
                  </div>

                  <div>
                    <span className="text-xs font-mono uppercase text-slate-400 block font-semibold mb-1">
                      7. Reasoning & Justification
                    </span>
                    <p className="text-sm text-slate-200 leading-relaxed">
                      {report.reasoning}
                    </p>
                  </div>
                </div>

                {/* Section 8 & 9: Technical Approach & Expected Contribution */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <span className="text-xs font-mono uppercase text-cyan-400 block font-semibold flex items-center space-x-1.5">
                      <Wrench className="w-3.5 h-3.5" />
                      <span>8. Technical Approach</span>
                    </span>
                    <p className="text-sm text-slate-200 leading-relaxed">
                      {report.technical_approach}
                    </p>
                  </div>

                  <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                    <span className="text-xs font-mono uppercase text-emerald-400 block font-semibold flex items-center space-x-1.5">
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>9. Expected Contribution</span>
                    </span>
                    <p className="text-sm text-slate-200 leading-relaxed">
                      {report.expected_contribution}
                    </p>
                  </div>
                </div>

                {/* Section 10: Feasibility Assessment */}
                {report.feasibility && (
                  <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono uppercase text-slate-400 font-semibold flex items-center space-x-1.5">
                        <Gauge className="w-3.5 h-3.5 text-amber-400" />
                        <span>10. Feasibility Assessment</span>
                      </span>
                      <span className="text-xs font-mono text-amber-400 font-bold">
                        Score: {report.feasibility.feasibility_score * 100}%
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                      <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs">
                        <span className="text-slate-400 block mb-1 font-mono">Estimated Timeline</span>
                        <span className="text-white font-medium">{report.feasibility.estimated_timeline || 'N/A'}</span>
                      </div>
                      
                      <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs">
                        <span className="text-slate-400 block mb-1 font-mono">Technical Barriers</span>
                        <ul className="list-disc list-inside text-slate-300 space-y-0.5">
                          {report.feasibility.technical_barriers?.map((b, i) => (
                            <li key={i}>{b}</li>
                          )) || <li>None recorded</li>}
                        </ul>
                      </div>

                      <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs">
                        <span className="text-slate-400 block mb-1 font-mono">Required Resources</span>
                        <ul className="list-disc list-inside text-slate-300 space-y-0.5">
                          {report.feasibility.required_resources?.map((r, i) => (
                            <li key={i}>{r}</li>
                          )) || <li>Standard compute</li>}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}

                {/* Section 11 & 12: Prior-Art Assessment & Differentiation Panel */}
                <div className="p-6 rounded-xl bg-slate-950 border border-slate-800 space-y-4">
                  <div className="flex items-center space-x-2 text-xs font-mono uppercase text-amber-400 font-semibold">
                    <GitCompare className="w-4 h-4" />
                    <span>11 & 12. Prior-Art Assessment & Differentiation</span>
                  </div>

                  {report.prior_art_assessment && (
                    <div className="space-y-3">
                      <p className="text-sm text-slate-300">
                        {report.prior_art_assessment.reasoning}
                      </p>

                      {report.differentiation && report.differentiation.length > 0 && (
                        <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 space-y-2">
                          <span className="text-xs font-mono uppercase text-indigo-400 block">
                            Key Differentiating Factors
                          </span>
                          <ul className="list-disc list-inside text-xs text-slate-200 space-y-1">
                            {report.differentiation.map((d, i) => (
                              <li key={i}>{d}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Section 13 & 14: Supporting Papers & Grounded Evidence */}
                <div className="space-y-4 pt-4 border-t border-slate-800">
                  <span className="text-xs font-mono uppercase text-slate-400 block font-semibold">
                    13 & 14. Ground Evidence & Supporting Papers ({report.supporting_evidence?.length || 0} items)
                  </span>

                  <div className="space-y-3">
                    {report.supporting_evidence?.map((ev) => (
                      <div
                        key={ev.statement_id}
                        onClick={() => onInspectEvidence(ev.statement_id)}
                        className="p-4 rounded-xl bg-slate-950 border border-slate-800 hover:border-amber-600/60 cursor-pointer transition flex items-start justify-between gap-4 group"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2 text-xs">
                            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800">
                              {ev.statement_type}
                            </span>
                            <span className="text-slate-400 font-medium">
                              {ev.paper_title || `Paper ID: ${ev.paper_id}`}
                            </span>
                            {ev.page_start !== null && ev.page_start !== undefined && (
                              <span className="text-slate-500 font-mono">
                                p.{ev.page_start}
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-slate-200 group-hover:text-amber-200 leading-relaxed italic">
                            "{ev.statement_text}"
                          </p>
                        </div>
                        <button className="text-xs font-mono text-amber-400 shrink-0 group-hover:underline">
                          Trace Evidence →
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            )}
          </div>

        </div>
      )}

    </div>
  );
};
