import React from 'react';
import { FlaskConical, Lock, ShieldCheck } from 'lucide-react';

export const EvaluationView: React.FC = () => {
  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
            <FlaskConical className="w-6 h-6 text-slate-400" />
            <span>Evaluation Benchmarks</span>
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Systematic experimental evaluation & statistical baseline benchmarks.
          </p>
        </div>
      </div>

      <div className="p-16 text-center rounded-2xl bg-slate-900/40 border border-slate-800 space-y-4 max-w-2xl mx-auto my-12">
        <div className="w-14 h-14 rounded-2xl bg-slate-800 text-slate-400 flex items-center justify-center mx-auto border border-slate-700">
          <Lock className="w-7 h-7 text-indigo-400" />
        </div>

        <div className="space-y-2">
          <h2 className="text-lg font-bold text-white">Phase 7 Experimental Benchmarking</h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Evaluation experiments, human expert pairwise comparisons, and temporal baseline tracking are scheduled for Phase 7 implementation.
          </p>
        </div>

        <div className="pt-4 flex justify-center">
          <div className="px-4 py-2 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 font-mono flex items-center space-x-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>No fake frontend evaluation data generated</span>
          </div>
        </div>
      </div>
    </div>
  );
};
