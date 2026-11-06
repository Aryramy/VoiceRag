"use client";

import { useState, useCallback } from "react";

export interface SessionState {
  token: string | null;
  url: string | null;
  isLoading: boolean;
  error: string | null;
}

export function useLiveKitSession() {
  const [session, setSession] = useState<SessionState>({
    token: null,
    url: null,
    isLoading: false,
    error: null,
  });

  const connect = useCallback(async (roomName: string = "voice-room-1", username?: string) => {
    setSession((prev) => ({ ...prev, isLoading: true, error: null }));
    try {
      const uname = username || `user-${Math.floor(Math.random() * 10000)}`;
      const res = await fetch(`/api/token?room=${encodeURIComponent(roomName)}&username=${encodeURIComponent(uname)}`);
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || `Failed to fetch join token (${res.status})`);
      }

      const data = await res.json();
      setSession({
        token: data.token,
        url: data.url,
        isLoading: false,
        error: null,
      });
    } catch (err: any) {
      setSession({
        token: null,
        url: null,
        isLoading: false,
        error: err.message || "Failed to establish voice session",
      });
    }
  }, []);

  const disconnect = useCallback((reason?: string) => {
    setSession((prev) => ({
      token: null,
      url: null,
      isLoading: false,
      error: reason || prev.error || null,
    }));
  }, []);

  const setError = useCallback((errMsg: string) => {
    setSession((prev) => ({
      ...prev,
      token: null,
      url: null,
      isLoading: false,
      error: errMsg,
    }));
  }, []);

  return {
    ...session,
    connect,
    disconnect,
    setError,
  };
}

