import React, { useState } from 'react';
import { UnifiedVerificationResponse } from '../types';

interface RawPayloadInspectorProps {
  result: UnifiedVerificationResponse;
}

export const RawPayloadInspector: React.FC<RawPayloadInspectorProps> = ({ result }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'all' | 'raw_state' | 'verdict' | 'fusion'>('all');

  const jsonString = React.useMemo(() => {
    let targetData: any = result;
    if (activeTab === 'raw_state') targetData = result.raw || {};
    else if (activeTab === 'verdict') targetData = result.raw?.final_verdict || result.factual_verification;
    else if (activeTab === 'fusion') targetData = result.raw?.source_agreement || {};
    return JSON.stringify(targetData, null, 2);
  }, [result, activeTab]);

  const handleCopy = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-surface-container rounded-xl p-5 border border-outline-variant/50 shadow-md flex flex-col gap-3">
      {/* Header Bar */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined text-[18px] text-outline">data_object</span>
          <span className="text-xs font-mono uppercase tracking-widest text-on-surface-variant font-semibold">
            Raw Forensic State & Payload Inspector
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-outline">
            {isOpen ? 'COLLAPSE' : 'EXPAND TO AUDIT RAW DATA'}
          </span>
          <span className="material-symbols-outlined text-[16px] text-outline">
            {isOpen ? 'expand_less' : 'expand_more'}
          </span>
        </div>
      </div>

      {isOpen && (
        <div className="flex flex-col gap-3 pt-2 border-t border-outline-variant/30">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1 font-mono text-[10px]">
              <button
                type="button"
                onClick={() => setActiveTab('all')}
                className={`px-2 py-0.5 rounded border ${
                  activeTab === 'all'
                    ? 'bg-surface-container-high text-primary border-primary/40 font-semibold'
                    : 'bg-surface-container-low text-outline border-outline-variant/20'
                }`}
              >
                Complete Unified Payload
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('raw_state')}
                className={`px-2 py-0.5 rounded border ${
                  activeTab === 'raw_state'
                    ? 'bg-surface-container-high text-primary border-primary/40 font-semibold'
                    : 'bg-surface-container-low text-outline border-outline-variant/20'
                }`}
              >
                Raw Pipeline State
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('verdict')}
                className={`px-2 py-0.5 rounded border ${
                  activeTab === 'verdict'
                    ? 'bg-surface-container-high text-primary border-primary/40 font-semibold'
                    : 'bg-surface-container-low text-outline border-outline-variant/20'
                }`}
              >
                Verdict Dict
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('fusion')}
                className={`px-2 py-0.5 rounded border ${
                  activeTab === 'fusion'
                    ? 'bg-surface-container-high text-primary border-primary/40 font-semibold'
                    : 'bg-surface-container-low text-outline border-outline-variant/20'
                }`}
              >
                Source Agreement
              </button>
            </div>

            <button
              type="button"
              onClick={handleCopy}
              className="px-2.5 py-1 rounded bg-surface-container-high hover:bg-surface-container-highest text-primary font-mono text-[10px] flex items-center gap-1 border border-outline-variant/30 transition-colors"
            >
              <span className="material-symbols-outlined text-[13px]">
                {copied ? 'check' : 'content_copy'}
              </span>
              <span>{copied ? 'Copied' : 'Copy JSON'}</span>
            </button>
          </div>

          <pre className="p-3.5 rounded-lg bg-surface-container-lowest border border-outline-variant/30 text-[11px] font-mono text-tertiary overflow-x-auto max-h-96 leading-relaxed select-text">
            {jsonString}
          </pre>
        </div>
      )}
    </div>
  );
};
