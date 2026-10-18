import React from 'react';
import { Sparkles, BookOpen, GitFork, Lightbulb, BarChart3, FlaskConical } from 'lucide-react';

export type TabView = 'overview' | 'literature' | 'landscape' | 'gaps' | 'opportunities' | 'evaluation';

interface NavbarProps {
  currentTab: TabView;
  onSelectTab: (tab: TabView) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, onSelectTab }) => {
  const navItems: { id: TabView; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'literature', label: 'Literature', icon: <BookOpen className="w-4 h-4" /> },
    { id: 'landscape', label: 'Research Landscape', icon: <GitFork className="w-4 h-4" /> },
    { id: 'gaps', label: 'Research Gaps', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'opportunities', label: 'Opportunity Studio', icon: <Lightbulb className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Evaluation', icon: <FlaskConical className="w-4 h-4 text-slate-400" /> },
  ];

  return (
    <header className="sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        
        {/* Brand logo */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={() => onSelectTab('overview')}>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg tracking-tight text-white">IdeaGen</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800/60 font-semibold">
                v2.0
              </span>
            </div>
            <span className="text-[11px] text-slate-400 block -mt-1 font-sans">
              Evidence-Grounded Research Discovery
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800/80">
          {navItems.map((item) => {
            const active = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  active
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Status indicator */}
        <div className="flex items-center space-x-2 text-xs text-emerald-400 bg-emerald-950/40 px-3 py-1.5 rounded-full border border-emerald-800/40 font-mono">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>Pipeline Ready</span>
        </div>

      </div>
    </header>
  );
};
