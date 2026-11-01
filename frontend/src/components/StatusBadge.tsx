import React from 'react';
import { FactualVerdict, SyntheticMediaSignal } from '../types';

interface StatusBadgeProps {
  type: 'factual' | 'synthetic' | 'visual' | 'tag';
  value: string | null | undefined;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ type, value, className = '' }) => {
  if (!value) {
    return (
      <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono uppercase bg-surface-container text-outline ${className}`}>
        Unavailable
      </span>
    );
  }

  const upper = value.toUpperCase();

  if (type === 'factual') {
    if (upper === 'REAL' || upper === 'SUPPORTED' || upper === 'TRUE') {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide bg-primary-container/20 text-primary border border-primary/30 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          SUPPORTED / REAL
        </span>
      );
    }
    if (upper === 'FAKE' || upper === 'CONTRADICTED' || upper === 'FALSE') {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide bg-secondary-container/40 text-on-secondary-container border border-secondary/40 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
          CONTRADICTED / FAKE
        </span>
      );
    }
    if (upper === 'MISLEADING') {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide bg-secondary-container/30 text-secondary border border-secondary/30 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
          MISLEADING
        </span>
      );
    }
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono tracking-wide bg-surface-container-high text-on-surface-variant border border-outline-variant ${className}`}>
        <span className="w-1.5 h-1.5 rounded-full bg-outline" />
        {upper}
      </span>
    );
  }

  if (type === 'synthetic') {
    if (upper === 'AI_GENERATED' || upper === 'AI-GENERATED' || upper === 'SYNTHETIC') {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide bg-secondary-container/40 text-on-secondary-container border border-secondary/40 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
          AI_GENERATED
        </span>
      );
    }
    if (upper === 'REAL' || upper === 'HUMAN_LIKE' || upper === 'AUTHENTIC') {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono font-medium tracking-wide bg-primary-container/20 text-primary border border-primary/30 ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-primary" />
          AUTHENTIC / REAL
        </span>
      );
    }
    if (upper === 'UNCERTAIN') {
      return (
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono tracking-wide bg-surface-container-high text-tertiary border border-outline-variant ${className}`}>
          <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
          UNCERTAIN
        </span>
      );
    }
    return (
      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono tracking-wide bg-surface-container-lowest text-outline border border-outline-variant ${className}`}>
        UNAVAILABLE
      </span>
    );
  }

  // Visual or generic tags
  if (upper === 'FAKE') {
    return (
      <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-secondary-container/30 text-secondary border border-secondary/30 ${className}`}>
        SUSPICIOUS / MANIPULATED
      </span>
    );
  }

  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-surface-container-high text-on-surface-variant ${className}`}>
      {value}
    </span>
  );
};
