"use client";

import React, { useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
} from "@livekit/components-react";
import { VoiceVisualizer } from "@/components/VoiceVisualizer";
import { Controls } from "@/components/Controls";
import { StatusBadge } from "@/components/StatusBadge";
import { DocumentUploader } from "@/components/DocumentUploader";
import { ChatBox } from "@/components/ChatBox";
import { Sidebar } from "@/components/Sidebar";
import { useLiveKitSession } from "@/hooks/useLiveKitSession";
import { useChatSession } from "@/hooks/useChatSession";
import { PhoneCall, Sparkles, BookOpen, ShieldCheck, Mic, MessageSquare, LayoutGrid } from "lucide-react";

export default function MultimodalWorkspacePage() {
  const { token, url, isLoading, error, connect, disconnect, setError } = useLiveKitSession();
  const {
    sessionId,
    documents,
    sessions,
    isLoadingDocs,
    isLoadingSessions,
    createNewSession,
    selectSession,
    refreshDocuments,
    deleteDocument,
    refreshSessions,
  } = useChatSession();

  const [activeTab, setActiveTab] = useState<"all" | "voice" | "chat">("all");

  const handleRoomError = (err: Error) => {
    console.error("LiveKit Room Error:", err);
    setError(
      `WebRTC Connection Error (${url}): ${err.message || "Failed to connect to LiveKit server. Please verify LiveKit server is running."}`
    );
  };

  const handleRoomDisconnected = () => {
    disconnect(
      "LiveKit session disconnected. (If unexpected, check if LiveKit server or cloud project is reachable)."
    );
  };

  return (
    <main className="flex flex-col min-h-screen p-4 sm:p-8 relative overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-slate-100">
      {/* Background ambient lighting */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <header className="w-full max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between border-b border-slate-800/80 pb-5 mb-6 gap-4 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-slate-950" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-100 tracking-tight">
              Multimodal Agentic RAG Workspace
            </h1>
            <p className="text-xs text-slate-400">
              LangGraph State Machine • Real-Time WebRTC Voice • Dynamic Ingestion • SQLite Memory
            </p>
          </div>
        </div>

        {/* View switcher and status */}
        <div className="flex items-center gap-3">
          <div className="flex items-center bg-slate-900/80 p-1 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => setActiveTab("all")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition ${
                activeTab === "all" ? "bg-indigo-600 text-white font-medium" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <LayoutGrid className="w-3.5 h-3.5" />
              <span>Workspace</span>
            </button>
            <button
              onClick={() => setActiveTab("voice")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition ${
                activeTab === "voice" ? "bg-indigo-600 text-white font-medium" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>Voice</span>
            </button>
            <button
              onClick={() => setActiveTab("chat")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition ${
                activeTab === "chat" ? "bg-indigo-600 text-white font-medium" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Chat</span>
            </button>
          </div>

          <StatusBadge isConnected={Boolean(token && url)} isLoading={isLoading} />
        </div>
      </header>

      {/* Main Workspace Grid */}
      <div className="w-full max-w-7xl mx-auto flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6 z-10 items-start">
        {/* Left Sidebar: Session History & Indexed Documents */}
        <div className="lg:col-span-3 w-full">
          <Sidebar
            currentSessionId={sessionId}
            sessions={sessions}
            documents={documents}
            isLoadingDocs={isLoadingDocs}
            isLoadingSessions={isLoadingSessions}
            onSelectSession={selectSession}
            onNewSession={createNewSession}
            onRefreshDocs={refreshDocuments}
            onDeleteDoc={deleteDocument}
          />
        </div>

        {/* Center / Right Content Area */}
        <div className="lg:col-span-9 w-full flex flex-col gap-6">
          {/* Top Document Uploader Bar */}
          <DocumentUploader
            onUploadComplete={() => {
              refreshDocuments();
            }}
          />

          {/* Multimodal Interaction Panes */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
            {/* Voice Assistant Module */}
            {(activeTab === "all" || activeTab === "voice") && (
              <div
                className={`${
                  activeTab === "all" ? "xl:col-span-6" : "xl:col-span-12"
                } flex flex-col items-center justify-center p-6 bg-slate-900/60 rounded-2xl border border-slate-800 backdrop-blur-md shadow-xl min-h-[460px]`}
              >
                <div className="w-full flex items-center justify-between pb-4 mb-4 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <Mic className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-sm font-semibold">LiveKit Voice Interface</h3>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">WebRTC 48kHz PCM</span>
                </div>

                {token && url ? (
                  <LiveKitRoom
                    token={token}
                    serverUrl={url}
                    connect={true}
                    audio={true}
                    video={false}
                    onError={handleRoomError}
                    onDisconnected={handleRoomDisconnected}
                    className="w-full flex flex-col items-center"
                  >
                    <RoomAudioRenderer />
                    <VoiceVisualizer />
                    <div className="mt-6">
                      <Controls onDisconnect={disconnect} />
                    </div>
                  </LiveKitRoom>
                ) : (
                  <div className="w-full flex flex-col items-center text-center p-4">
                    <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-4">
                      <PhoneCall className="w-7 h-7 text-indigo-400" />
                    </div>
                    <h2 className="text-lg font-bold text-slate-100 mb-1">
                      Voice Assistant Ready
                    </h2>
                    <p className="text-xs text-slate-400 max-w-sm mb-6 leading-relaxed">
                      Connect via WebRTC for real-time speech conversations grounded on your indexed documents with zero cloud lock-in.
                    </p>

                    {error && (
                      <div className="mb-4 p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs max-w-sm">
                        {error}
                      </div>
                    )}

                    <button
                      onClick={() =>
                        connect(`voice-room-${Math.floor(Math.random() * 10000)}`)
                      }
                      disabled={isLoading}
                      className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-indigo-600 hover:from-indigo-600 hover:to-indigo-700 text-white font-medium text-xs flex items-center gap-2 shadow-lg shadow-indigo-500/25 transition disabled:opacity-50"
                    >
                      <PhoneCall className="w-4 h-4" />
                      <span>{isLoading ? "Connecting WebRTC..." : "Start Voice Call"}</span>
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Text Chat Interface Module */}
            {(activeTab === "all" || activeTab === "chat") && (
              <div
                className={`${
                  activeTab === "all" ? "xl:col-span-6" : "xl:col-span-12"
                } w-full`}
              >
                <ChatBox
                  sessionId={sessionId}
                  onNewMessage={() => {
                    refreshSessions();
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature & Architecture highlights footer */}
      <footer className="w-full max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-4 pt-6 mt-8 border-t border-slate-800/80 z-10 text-xs text-slate-400">
        <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
          <BookOpen className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>Local Embedded Qdrant vector retrieval</span>
        </div>
        <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
          <Sparkles className="w-4 h-4 text-indigo-400 shrink-0" />
          <span>LangGraph rewrite & structured citations</span>
        </div>
        <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
          <ShieldCheck className="w-4 h-4 text-amber-400 shrink-0" />
          <span>Strict fallback guard & SQLite persistence</span>
        </div>
      </footer>
    </main>
  );
}
