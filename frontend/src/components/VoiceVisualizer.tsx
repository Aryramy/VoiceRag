"use client";

import React from "react";
import {
  useVoiceAssistant,
  BarVisualizer,
} from "@livekit/components-react";
import { Mic, Cpu, Volume2, Radio } from "lucide-react";

export function VoiceVisualizer() {
  const { state, audioTrack, agent } = useVoiceAssistant();

  const getStatusConfig = () => {
    switch (state) {
      case "listening":
        return {
          label: "Listening...",
          color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          icon: Mic,
          pulse: true,
        };
      case "thinking":
        return {
          label: "Thinking & Searching RAG...",
          color: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          icon: Cpu,
          pulse: true,
        };
      case "speaking":
        return {
          label: "Speaking",
          color: "bg-indigo-500/10 text-indigo-400 border-indigo-500/30",
          icon: Volume2,
          pulse: false,
        };
      default:
        return {
          label: "Connecting...",
          color: "bg-slate-500/10 text-slate-400 border-slate-500/30",
          icon: Radio,
          pulse: false,
        };
    }
  };

  const status = getStatusConfig();
  const StatusIcon = status.icon;

  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-6">
      {/* State Badge */}
      <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-sm font-medium ${status.color} ${status.pulse ? "animate-pulse" : ""}`}>
        <StatusIcon className="w-4 h-4" />
        <span>{status.label}</span>
      </div>

      {/* Dynamic Audio Spectrum Visualizer */}
      <div className="w-full max-w-md h-32 flex items-center justify-center bg-slate-900/50 rounded-2xl border border-slate-800 p-4">
        {audioTrack ? (
          <BarVisualizer
            state={state}
            barCount={24}
            trackRef={audioTrack}
            className="w-full h-full flex items-center justify-center gap-1.5"
          />
        ) : (
          <div className="text-slate-500 text-sm flex items-center gap-2">
            <Radio className="w-4 h-4 animate-spin" />
            <span>Waiting for audio stream...</span>
          </div>
        )}
      </div>

      {/* Agent Info */}
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
        <span>Connected to LiveKit WebRTC Agent</span>
      </div>
    </div>
  );
}

