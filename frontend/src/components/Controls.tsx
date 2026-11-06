"use client";

import React from "react";
import { useRoomContext, TrackToggle } from "@livekit/components-react";
import { Track } from "livekit-client";
import { Mic, MicOff, PhoneOff } from "lucide-react";

interface ControlsProps {
  onDisconnect?: () => void;
}

export function Controls({ onDisconnect }: ControlsProps) {
  const room = useRoomContext();

  const handleDisconnect = () => {
    if (room) {
      room.disconnect();
    }
    if (onDisconnect) {
      onDisconnect();
    }
  };

  return (
    <div className="flex items-center justify-center gap-4 p-4 bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800">
      <TrackToggle
        source={Track.Source.Microphone}
        className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 transition-all font-medium text-sm flex items-center gap-2"
      />
      <button
        onClick={handleDisconnect}
        className="px-4 py-2.5 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-400 border border-rose-500/30 transition-all font-medium text-sm flex items-center gap-2"
      >
        <PhoneOff className="w-4 h-4" />
        <span>End Call</span>
      </button>
    </div>
  );
}

