import React from 'react';

interface ConfidenceMeterProps {
  score: number | null; // 0.0 to 1.0 or percentage
  label?: string;
  variant?: 'primary' | 'secondary' | 'neutral';
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  score,
  label = 'Heuristic Weight',
  variant = 'secondary',
}) => {
  if (score === null || score === undefined || isNaN(score)) {
    return (
      <div className="flex flex-col gap-1 w-full opacity-60">
        <div className="flex justify-between font-label text-[10px] text-outline uppercase tracking-wider">
          <span>{label}</span>
          <span>Score Unavailable</span>
        </div>
        <div className="grid grid-cols-10 gap-1 h-2 w-full">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="bg-surface-container-highest rounded-sm" />
          ))}
        </div>
      </div>
    );
  }

  // Normalize score to 0..10
  const normalized = score > 1 ? score / 100 : score;
  const filledDashes = Math.round(Math.min(Math.max(normalized, 0), 1) * 10);

  const activeColor =
    variant === 'primary'
      ? 'bg-primary'
      : variant === 'secondary'
      ? 'bg-secondary'
      : 'bg-tertiary';

  const statusLabel =
    normalized >= 0.8
      ? 'High Confidence'
      : normalized >= 0.5
      ? 'Moderate Confidence'
      : 'Low / Uncertain';

  return (
    <div className="flex flex-col gap-1 w-full">
      <div className="flex justify-between font-label text-[10px] text-outline uppercase tracking-wider">
        <span>{label}</span>
        <span className={variant === 'secondary' ? 'text-secondary font-semibold' : 'text-primary font-semibold'}>
          {statusLabel} ({(normalized * 100).toFixed(1)}%)
        </span>
      </div>
      <div className="grid grid-cols-10 gap-1 h-2 w-full" role="meter" aria-valuenow={Math.round(normalized * 100)}>
        {Array.from({ length: 10 }).map((_, i) => (
          <div
            key={i}
            className={`rounded-sm transition-colors duration-200 ${
              i < filledDashes ? activeColor : 'bg-surface-container-highest'
            }`}
          />
        ))}
      </div>
    </div>
  );
};
