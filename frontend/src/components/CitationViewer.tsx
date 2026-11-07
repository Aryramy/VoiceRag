"use client";

import React, { useState, useEffect, useRef } from "react";
import { BookOpen, ChevronDown, ChevronUp, FileText, CheckCircle2 } from "lucide-react";

export interface Citation {
  chunk_id: string;
  source_file: string;
  text: string;
}

interface CitationViewerProps {
  citations: Citation[];
  defaultExpanded?: boolean;
  isExpanded?: boolean;
  onToggleExpanded?: () => void;
  highlightedChunkId?: string | null;
  onSelectCitation?: (chunkId: string) => void;
}

export function CitationViewer({
  citations,
  defaultExpanded = false,
  isExpanded: controlledExpanded,
  onToggleExpanded,
  highlightedChunkId,
  onSelectCitation,
}: CitationViewerProps) {
  const [internalExpanded, setInternalExpanded] = useState(defaultExpanded);
  const isExpanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded;
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const toggle = () => {
    if (onToggleExpanded) {
      onToggleExpanded();
    } else {
      setInternalExpanded(!internalExpanded);
    }
  };

  // Automatically scroll highlighted citation into view
  useEffect(() => {
    if (isExpanded && highlightedChunkId) {
      // Find key in refs matching chunk_id or variants
      const targetRef =
        itemRefs.current[highlightedChunkId] ||
        itemRefs.current[`chunk_${highlightedChunkId}`] ||
        itemRefs.current[highlightedChunkId.replace(/^chunk_/, "")];

      if (targetRef) {
        targetRef.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  }, [highlightedChunkId, isExpanded]);

  if (!citations || citations.length === 0) {
    return null;
  }

  const matchesHighlight = (chunkId: string) => {
    if (!highlightedChunkId) return false;
    return (
      chunkId === highlightedChunkId ||
      chunkId === `chunk_${highlightedChunkId}` ||
      chunkId.replace(/^chunk_/, "") === highlightedChunkId.replace(/^chunk_/, "")
    );
  };

  return (
    <div className="mt-2 w-full text-xs">
      <button
        type="button"
        onClick={toggle}
        className={`flex items-center gap-1.5 text-[11px] font-medium transition-all py-1.5 px-2.5 rounded-lg border ${highlightedChunkId
            ? "bg-indigo-600/20 text-indigo-200 border-indigo-500/50 shadow-sm shadow-indigo-500/20"
            : "bg-indigo-500/10 text-indigo-400 hover:text-indigo-300 border-indigo-500/20 hover:border-indigo-500/40"
          }`}
      >
        <BookOpen className="w-3.5 h-3.5 shrink-0" />
        <span>
          {citations.length} Grounded Source{citations.length > 1 ? "s" : ""}
        </span>
        {isExpanded ? (
          <ChevronUp className="w-3.5 h-3.5 ml-0.5" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 ml-0.5" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-2 p-3 bg-slate-950/90 rounded-xl border border-slate-800 text-[11px] animate-fadeIn shadow-xl max-h-72 overflow-y-auto">
          {citations.map((c, idx) => {
            const isTarget = matchesHighlight(c.chunk_id);
            return (
              <div
                key={`${c.chunk_id}-${idx}`}
                ref={(el) => {
                  itemRefs.current[c.chunk_id] = el;
                }}
                onClick={() => onSelectCitation && onSelectCitation(c.chunk_id)}
                className={`space-y-1.5 p-2.5 rounded-xl border transition-all duration-300 cursor-pointer ${isTarget
                    ? "border-indigo-500/90 bg-indigo-950/60 ring-2 ring-indigo-400/80 shadow-md shadow-indigo-500/20 scale-[1.01]"
                    : "border-slate-800/80 bg-slate-900/40 hover:border-slate-700/80 hover:bg-slate-900/70"
                  }`}
              >
                <div className="flex items-center justify-between text-slate-400 font-mono text-[10px]">
                  <div className="flex items-center gap-1.5 truncate max-w-[240px]">
                    <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 shrink-0">
                      [{idx + 1}]
                    </span>
                    <span className="flex items-center gap-1 text-indigo-400 font-semibold truncate">
                      <FileText className="w-3 h-3 shrink-0" />
                      <span className="truncate" title={c.source_file}>{c.source_file}</span>
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {isTarget && (
                      <span className="inline-flex items-center gap-1 text-[10px] font-sans font-medium text-indigo-300 bg-indigo-500/30 px-1.5 py-0.5 rounded animate-pulse">
                        <CheckCircle2 className="w-2.5 h-2.5" />
                        <span>Source #{idx + 1}</span>
                      </span>
                    )}
                    <span className="text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                      [{c.chunk_id}]
                    </span>
                  </div>
                </div>
                <p className="text-slate-300 italic bg-slate-950/60 p-2 rounded-lg border border-slate-800/50 text-[11px] leading-relaxed">
                  "{c.text}"
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
