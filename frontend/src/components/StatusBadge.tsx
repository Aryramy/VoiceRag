"use client";

import React from "react";

interface StatusBadgeProps {
  isConnected?: boolean;
  isLoading?: boolean;
}

export function StatusBadge({ isConnected, isLoading }: StatusBadgeProps) {
  if (isLoading) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border bg-amber-500/10 text-amber-400 border-amber-500/20">
        <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
        <span>Connecting...</span>
      </div>
    );
  }

  if (isConnected) {
    return (
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border bg-emerald-500/10 text-emerald-400 border-emerald-500/20">
        <span className="w-2 h-2 rounded-full bg-emerald-400" />
        <span>LiveKit WebRTC Active</span>
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border bg-slate-500/10 text-slate-400 border-slate-500/20">
      <span className="w-2 h-2 rounded-full bg-slate-400" />
      <span>Standby</span>
    </div>
  );
}
