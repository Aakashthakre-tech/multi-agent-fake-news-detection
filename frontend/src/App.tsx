import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { BottomNav } from './components/BottomNav';
import { VerifyPage } from './features/verify/VerifyPage';
import { SystemOverviewPage } from './features/system-overview/SystemOverviewPage';
import { fetchHealth } from './services/api';
import { SystemHealthResponse, UnifiedVerificationResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'verify' | 'overview'>('verify');
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);
  const [investigationCount, setInvestigationCount] = useState<number>(0);

  // Check health on load
  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      try {
        const h = await fetchHealth();
        if (isMounted) {
          setHealth(h);
          setApiHealthy(true);
        }
      } catch (err) {
        if (isMounted) {
          console.warn('API health check error:', err);
          setApiHealthy(false);
        }
      }
    };

    check();
    const interval = setInterval(check, 30000); // 30s poll
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleAnalysisComplete = (_result: UnifiedVerificationResponse) => {
    setInvestigationCount((prev) => prev + 1);
  };

  return (
    <div className="min-h-screen bg-surface text-on-surface flex flex-col font-sans selection:bg-primary-container selection:text-on-primary">
      {/* Top Fixed Header */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        investigationCount={investigationCount}
        apiHealthy={apiHealthy}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 w-full pt-16 flex flex-col items-center">
        {activeTab === 'verify' ? (
          <VerifyPage
            health={health}
            onAnalysisComplete={handleAnalysisComplete}
          />
        ) : (
          <SystemOverviewPage />
        )}
      </main>

      {/* Mobile Bottom Navigation Bar */}
      <BottomNav activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  );
};

export default App;
