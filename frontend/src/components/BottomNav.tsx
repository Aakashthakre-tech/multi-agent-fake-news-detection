import React from 'react';

interface BottomNavProps {
  activeTab: 'verify' | 'overview';
  onTabChange: (tab: 'verify' | 'overview') => void;
}

export const BottomNav: React.FC<BottomNavProps> = ({ activeTab, onTabChange }) => {
  return (
    <nav className="fixed bottom-0 inset-x-0 z-50 pb-safe bg-surface/95 backdrop-blur-xl border-t border-outline-variant/40 shadow-[0_-2px_12px_rgba(0,0,0,0.45)] sm:hidden">
      <div className="grid grid-cols-2 items-center h-16 max-w-md mx-auto px-4">
        <button
          type="button"
          onClick={() => onTabChange('verify')}
          className={`min-h-[44px] h-full flex flex-col items-center justify-center gap-0.5 transition-colors ${
            activeTab === 'verify'
              ? 'text-primary font-medium border-t-2 border-primary -mt-[1px]'
              : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-[20px]">verified</span>
          <span className="text-[11px] tracking-wider font-mono uppercase">Verify</span>
        </button>

        <button
          type="button"
          onClick={() => onTabChange('overview')}
          className={`min-h-[44px] h-full flex flex-col items-center justify-center gap-0.5 transition-colors ${
            activeTab === 'overview'
              ? 'text-primary font-medium border-t-2 border-primary -mt-[1px]'
              : 'text-on-surface-variant hover:text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-[20px]">account_tree</span>
          <span className="text-[11px] tracking-wider font-mono uppercase">Overview</span>
        </button>
      </div>
    </nav>
  );
};
