"use client";

import React, { useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
} from "@livekit/components-react";
import { VoiceVisualizer } from "@/components/VoiceVisualizer";
import { Controls } from "@/components/Controls";
import { StatusBadge } from "@/components/StatusBadge";
import { useLiveKitSession } from "@/hooks/useLiveKitSession";
import { PhoneCall, Sparkles, BookOpen, ShieldCheck } from "lucide-react";

export default function VoiceAssistantPage() {
  const { token, url, isLoading, error, connect, disconnect, setError } = useLiveKitSession();
  const [roomName, setRoomName] = useState("voice-room-1");

  const handleRoomError = (err: Error) => {
    console.error("LiveKit Room Error:", err);
    setError(`WebRTC Connection Error (${url}): ${err.message || "Failed to connect to LiveKit server. Please verify LiveKit server is running."}`);
  };

  const handleRoomDisconnected = () => {
    disconnect("LiveKit session disconnected. (If unexpected, check if LiveKit server or cloud project is reachable).");
  };

  return (
    <main className="flex flex-col min-h-screen items-center justify-between p-6 sm:p-12 relative overflow-hidden bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black">
      {/* Background ambient lighting */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <header className="w-full max-w-4xl flex items-center justify-between border-b border-slate-800/80 pb-6 z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles className="w-5 h-5 text-slate-950" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-100 tracking-tight">Agentic Voice AI</h1>
            <p className="text-xs text-slate-400">LangGraph RAG + LiveKit WebRTC</p>
          </div>
        </div>
        <StatusBadge isConnected={Boolean(token && url)} isLoading={isLoading} />
      </header>

      {/* Main Interaction Area */}
      <section className="w-full max-w-2xl my-auto py-8 z-10 flex flex-col items-center">
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
            {/* Live Audio In/Out Renderer */}
            <RoomAudioRenderer />

            {/* Real-time spectrum visualizer and state indicator */}
            <VoiceVisualizer />

            {/* Interaction controls */}
            <div className="mt-6">
              <Controls onDisconnect={disconnect} />
            </div>
          </LiveKitRoom>
        ) : (
          <div className="w-full flex flex-col items-center text-center p-8 bg-slate-900/40 rounded-3xl border border-slate-800/80 backdrop-blur-xl shadow-2xl">
            <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-6">
              <PhoneCall className="w-8 h-8 text-indigo-400" />
            </div>
            <h2 className="text-2xl font-bold text-slate-100 mb-2">Connect to Knowledge Voice Assistant</h2>
            <p className="text-sm text-slate-400 max-w-md mb-8">
              Experience real-time conversational voice search powered by LangGraph, user-space embedded Qdrant, and sub-second WebRTC streaming.
            </p>

            {error && (
              <div className="mb-6 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs max-w-md">
                {error}
              </div>
            )}

            <button
              onClick={() => connect(`voice-room-${Math.floor(Math.random() * 10000)}`)}
              disabled={isLoading}
              className="px-8 py-3.5 rounded-2xl bg-gradient-to-r from-indigo-500 to-indigo-600 hover:from-indigo-600 hover:to-indigo-700 text-white font-medium text-sm flex items-center gap-2.5 shadow-xl shadow-indigo-500/25 transition-all disabled:opacity-50"
            >
              <PhoneCall className="w-4 h-4" />
              <span>{isLoading ? "Starting Session..." : "Start Voice Call"}</span>
            </button>
          </div>
        )}
      </section>

      {/* Feature & Architecture highlights footer */}
      <footer className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-3 gap-4 pt-6 border-t border-slate-800/80 z-10 text-xs text-slate-400">
        <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
          <BookOpen className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>Local Embedded Qdrant vector retrieval</span>
        </div>
        <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
          <Sparkles className="w-4 h-4 text-indigo-400 shrink-0" />
          <span>LangGraph rewrite & citation pipeline</span>
        </div>
        <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-900/30 border border-slate-800/50">
          <ShieldCheck className="w-4 h-4 text-amber-400 shrink-0" />
          <span>Strict fallback guard for grounding</span>
        </div>
      </footer>
    </main>
  );
}

