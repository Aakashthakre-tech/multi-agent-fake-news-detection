import React from 'react';

interface HeaderProps {
  activeTab: 'verify' | 'overview';
  onTabChange: (tab: 'verify' | 'overview') => void;
  investigationCount?: number;
  apiHealthy?: boolean | null;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  investigationCount = 1,
  apiHealthy = true,
}) => {
  return (
    <header className="fixed top-0 inset-x-0 z-50 bg-surface/90 backdrop-blur-xl shadow-[0_1px_8px_rgba(0,0,0,0.45)] border-b border-outline-variant/40 pt-safe">
      <div className="max-w-7xl mx-auto h-16 px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-4">
        {/* Brand & Wordmark */}
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-surface-container-high border border-outline-variant/60 flex items-center justify-center flex-shrink-0 text-primary">
            <span className="material-symbols-outlined text-[20px]">biotech</span>
          </div>
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="font-serif text-lg sm:text-xl text-tertiary tracking-tight font-medium">
                TruthLens
              </span>
              <span className="text-[10px] font-mono text-primary font-semibold uppercase px-1 py-0.2 rounded bg-primary/10 border border-primary/20">
                AI
              </span>
            </div>
            <span className="text-[9px] font-mono text-on-surface-variant uppercase tracking-widest truncate">
              FORENSIC DESK // v3.4-LIVE
            </span>
          </div>
        </div>

        {/* Desktop / Tablet Navigation Controls */}
        <nav className="hidden sm:flex items-center bg-surface-container-lowest p-1 rounded-xl border border-outline-variant/40">
          <button
            type="button"
            onClick={() => onTabChange('verify')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
              activeTab === 'verify'
                ? 'bg-surface-container-high text-primary font-semibold shadow-sm border border-outline-variant/50'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-[16px]">verified</span>
            <span>Intake & Forensics</span>
          </button>
          <button
            type="button"
            onClick={() => onTabChange('overview')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
              activeTab === 'overview'
                ? 'bg-surface-container-high text-primary font-semibold shadow-sm border border-outline-variant/50'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-[16px]">account_tree</span>
            <span>System Architecture</span>
          </button>
        </nav>

        {/* Actions & Live Telemetry */}
        <div className="flex items-center gap-2.5 flex-shrink-0">
          {/* Backend API Health Status */}
          <div className="hidden md:flex items-center gap-1.5 px-2 py-1 rounded bg-surface-container text-[10px] font-mono border border-outline-variant/30">
            <span
              className={`w-2 h-2 rounded-full ${
                apiHealthy === true
                  ? 'bg-primary animate-pulse'
                  : apiHealthy === false
                  ? 'bg-error'
                  : 'bg-secondary animate-pulse'
              }`}
            />
            <span className="text-on-surface-variant">
              {apiHealthy === true ? 'API ONLINE' : apiHealthy === false ? 'API OFFLINE' : 'CHECKING...'}
            </span>
          </div>

          {/* Investigations Queue Pill */}
          <div
            title="Active Investigations Ledger"
            className="h-9 px-2.5 flex items-center justify-center rounded-lg bg-surface-container-high border border-outline-variant/40 text-on-surface"
          >
            <div className="flex items-center gap-1.5">
              <span className="material-symbols-outlined text-[18px] text-secondary">fact_check</span>
              <span className="text-xs font-mono text-tertiary font-semibold">{investigationCount}</span>
            </div>
          </div>

          {/* User Avatar */}
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0 shadow-sm border border-primary/50">
            <span className="material-symbols-outlined text-on-primary text-[18px]">person</span>
          </div>
        </div>
      </div>
    </header>
  );
};
