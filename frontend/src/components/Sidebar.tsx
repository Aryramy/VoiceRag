"use client";

import React from "react";
import {
  FileText,
  Plus,
  MessageSquare,
  RefreshCw,
  Database,
  CheckCircle2,
  Clock,
  AlertCircle,
  Layers,
  Trash2,
} from "lucide-react";
import { DocumentInfo, SessionInfo } from "@/hooks/useChatSession";

interface SidebarProps {
  currentSessionId: string;
  sessions: SessionInfo[];
  documents: DocumentInfo[];
  isLoadingDocs: boolean;
  isLoadingSessions: boolean;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  onRefreshDocs: () => void;
  onDeleteDoc?: (filename: string) => void;
}

export function Sidebar({
  currentSessionId,
  sessions,
  documents,
  isLoadingDocs,
  isLoadingSessions,
  onSelectSession,
  onNewSession,
  onRefreshDocs,
  onDeleteDoc,
}: SidebarProps) {
  return (
    <aside className="w-full lg:w-72 flex flex-col gap-5 bg-slate-900/60 p-4 rounded-2xl border border-slate-800 backdrop-blur-md">
      {/* Session Controls */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <MessageSquare className="w-3.5 h-3.5 text-indigo-400" />
            <span>Chat Sessions</span>
          </h3>
          <button
            onClick={onNewSession}
            className="flex items-center gap-1 text-[11px] bg-indigo-600 hover:bg-indigo-500 text-white px-2.5 py-1 rounded-lg transition shadow-sm font-medium"
          >
            <Plus className="w-3 h-3" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Sessions List */}
        <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
          {isLoadingSessions && sessions.length === 0 ? (
            <div className="text-[11px] text-slate-500 py-2 text-center flex items-center justify-center gap-2">
              <RefreshCw className="w-3 h-3 animate-spin text-indigo-400" />
              <span>Loading sessions...</span>
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-[11px] text-slate-500 py-2 text-center">No previous sessions</div>
          ) : (
            sessions.map((s) => {
              const isActive = s.session_id === currentSessionId;
              return (
                <button
                  key={s.session_id}
                  onClick={() => onSelectSession(s.session_id)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-left transition ${
                    isActive
                      ? "bg-indigo-600/20 text-indigo-200 border border-indigo-500/30"
                      : "bg-slate-950/40 text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-slate-850"
                  }`}
                >
                  <span className="truncate font-mono text-[11px]">
                    {s.title || `Session ${s.session_id.slice(0, 8)}`}
                  </span>
                  {isActive && <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />}
                </button>
              );
            })
          )}
        </div>
      </div>

      <div className="border-t border-slate-800/80" />

      {/* Indexed Documents Section */}
      <div className="space-y-3 flex-1 flex flex-col">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <Database className="w-3.5 h-3.5 text-emerald-400" />
            <span>Indexed KB ({documents.length})</span>
          </h3>
          <button
            onClick={onRefreshDocs}
            disabled={isLoadingDocs}
            title="Refresh documents"
            className="text-slate-400 hover:text-slate-200 p-1 rounded-md transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoadingDocs ? "animate-spin text-indigo-400" : ""}`} />
          </button>
        </div>

        <div className="space-y-2 flex-1 max-h-64 overflow-y-auto pr-1">
          {documents.length === 0 ? (
            <div className="p-3 bg-slate-950/40 rounded-xl border border-slate-800 text-[11px] text-slate-500 text-center">
              No custom files indexed yet. Upload a document to get started.
            </div>
          ) : (
            documents.map((doc, idx) => (
              <div
                key={`${doc.filename}-${idx}`}
                className="p-2.5 bg-slate-950/50 rounded-xl border border-slate-800/90 text-[11px] space-y-1.5 hover:border-slate-700/80 transition"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 font-medium text-slate-200 truncate">
                    <FileText className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                    <span className="truncate" title={doc.filename}>{doc.filename}</span>
                  </div>
                  {doc.status === "indexed" ? (
                    <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 shrink-0">
                      <CheckCircle2 className="w-2.5 h-2.5" />
                      <span>indexed</span>
                    </span>
                  ) : doc.status === "processing" ? (
                    <span className="inline-flex items-center gap-1 text-[10px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20 shrink-0">
                      <Clock className="w-2.5 h-2.5 animate-spin" />
                      <span>indexing</span>
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[10px] text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded border border-rose-500/20 shrink-0">
                      <AlertCircle className="w-2.5 h-2.5" />
                      <span className="truncate max-w-[120px]" title={doc.status}>{doc.status}</span>
                    </span>
                  )}
                </div>

                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                  <span className="flex items-center gap-1">
                    <Layers className="w-3 h-3 text-slate-500" />
                    {doc.chunk_count} chunk{doc.chunk_count === 1 ? "" : "s"}
                  </span>
                  <div className="flex items-center gap-2">
                    <span>{doc.uploaded_at ? doc.uploaded_at.split(" ")[0] : ""}</span>
                    {onDeleteDoc && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteDoc(doc.filename);
                        }}
                        title={`Delete ${doc.filename}`}
                        className="text-slate-500 hover:text-rose-400 p-0.5 rounded transition"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
}
