"use client";

import { useState, useEffect, useCallback } from "react";

export interface DocumentInfo {
  filename: string;
  chunk_count: number;
  status: string;
  uploaded_at: string;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  title?: string;
}

export function useChatSession() {
  const [sessionId, setSessionId] = useState<string>("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState<boolean>(false);
  const [isLoadingSessions, setIsLoadingSessions] = useState<boolean>(false);

  // Initialize or load session from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("rag_chat_session_id");
    if (saved) {
      setSessionId(saved);
    } else {
      const newId = `session-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
      setSessionId(newId);
      localStorage.setItem("rag_chat_session_id", newId);
    }
  }, []);

  const createNewSession = useCallback(() => {
    const newId = `session-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;
    setSessionId(newId);
    localStorage.setItem("rag_chat_session_id", newId);
    refreshSessions();
    return newId;
  }, []);

  const selectSession = useCallback((id: string) => {
    setSessionId(id);
    localStorage.setItem("rag_chat_session_id", id);
  }, []);

  const refreshDocuments = useCallback(async () => {
    setIsLoadingDocs(true);
    try {
      const res = await fetch("/api/upload");
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.error("Failed to load documents:", e);
    } finally {
      setIsLoadingDocs(false);
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    setIsLoadingSessions(true);
    try {
      const res = await fetch("/api/chat");
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions || []);
      }
    } catch (e) {
      console.error("Failed to load sessions:", e);
    } finally {
      setIsLoadingSessions(false);
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
    refreshSessions();
  }, [refreshDocuments, refreshSessions]);

  return {
    sessionId,
    documents,
    sessions,
    isLoadingDocs,
    isLoadingSessions,
    createNewSession,
    selectSession,
    refreshDocuments,
    refreshSessions,
  };
}

