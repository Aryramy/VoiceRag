"use client";

import React, { useState, useEffect, useRef } from "react";
import { Send, Bot, User, Sparkles } from "lucide-react";
import { CitationViewer, Citation } from "./CitationViewer";

export interface Message {
  sender: "user" | "agent";
  content: string;
  citations?: Citation[];
  timestamp?: string;
}

interface ChatBoxProps {
  sessionId: string;
  onNewMessage?: () => void;
}

// Regex to catch tags like [chunk_uuid], [chunk_1], [chunk_xxx]
const CITATION_TAG_REGEX = /(\[(?:chunk_[^\]\s]+)\])/g;

function ChatMessageItem({ msg, idx }: { msg: Message; idx: number }) {
  const [isAccordionOpen, setIsAccordionOpen] = useState(false);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);

  const handleBadgeClick = (chunkId: string) => {
    // If clicking the currently highlighted chunk, keep open or re-focus
    setIsAccordionOpen(true);
    setHighlightedChunkId(chunkId);
  };

  const renderMessageContent = (content: string, citations?: Citation[]) => {
    if (msg.sender === "user") {
      return content;
    }

    // Split content by citation tag regex: [chunk_xxx]
    const parts = content.split(CITATION_TAG_REGEX);
    let fallbackCounter = 1;

    return parts.map((part, partIdx) => {
      const match = part.match(/^\[(chunk_[^\]\s]+)\]$/);
      if (match) {
        const rawChunkId = match[1];

        // Match against msg.citations array to find the correct 1-based index
        const citationIdx = citations
          ? citations.findIndex(
            (c) =>
              c.chunk_id === rawChunkId ||
              c.chunk_id === rawChunkId.replace(/^chunk_/, "") ||
              `chunk_${c.chunk_id}` === rawChunkId
          )
          : -1;

        const badgeNumber = citationIdx !== -1 ? citationIdx + 1 : fallbackCounter++;
        const matchedCitation = citationIdx !== -1 && citations ? citations[citationIdx] : null;

        const isSelected =
          highlightedChunkId !== null &&
          (rawChunkId === highlightedChunkId ||
            rawChunkId.replace(/^chunk_/, "") === highlightedChunkId.replace(/^chunk_/, "") ||
            `chunk_${rawChunkId}` === highlightedChunkId);

        return (
          <button
            key={partIdx}
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleBadgeClick(rawChunkId);
            }}
            className={`inline-flex items-center justify-center font-mono text-[10px] font-bold px-1.5 py-0.5 mx-0.5 rounded-md transition-all duration-200 cursor-pointer align-baseline select-none shadow-sm ${isSelected
                ? "bg-indigo-500 text-white ring-2 ring-indigo-400 scale-105 shadow-indigo-500/30"
                : "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500/35 hover:text-white hover:border-indigo-400 hover:scale-105"
              }`}
            title={`View source [${badgeNumber}]: ${matchedCitation?.source_file || rawChunkId}`}
          >
            [{badgeNumber}]
          </button>
        );
      }

      return <span key={partIdx}>{part}</span>;
    });
  };

  return (
    <div className={`flex flex-col ${msg.sender === "user" ? "items-end" : "items-start"}`}>
      <div className="flex items-center gap-1.5 mb-1 px-1 text-[10px] text-slate-500">
        {msg.sender === "user" ? (
          <>
            <span>You</span>
            <User className="w-3 h-3 text-indigo-400" />
          </>
        ) : (
          <>
            <Bot className="w-3 h-3 text-emerald-400" />
            <span>Agent</span>
          </>
        )}
      </div>

      <div
        className={`max-w-[85%] rounded-2xl p-3.5 text-xs leading-relaxed whitespace-pre-wrap ${msg.sender === "user"
            ? "bg-indigo-600 text-white rounded-br-none shadow-md shadow-indigo-600/10"
            : "bg-slate-800/90 text-slate-200 border border-slate-700/60 rounded-bl-none shadow-md"
          }`}
      >
        {renderMessageContent(msg.content, msg.citations)}
      </div>

      {/* Render Grounded Citations Accordion */}
      {msg.citations && msg.citations.length > 0 && (
        <div className="max-w-[85%] w-full">
          <CitationViewer
            citations={msg.citations}
            isExpanded={isAccordionOpen}
            onToggleExpanded={() => setIsAccordionOpen(!isAccordionOpen)}
            highlightedChunkId={highlightedChunkId}
            onSelectCitation={(chunkId) =>
              setHighlightedChunkId(chunkId === highlightedChunkId ? null : chunkId)
            }
          />
        </div>
      )}
    </div>
  );
}

export function ChatBox({ sessionId, onNewMessage }: ChatBoxProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Load chat history when sessionId changes
  useEffect(() => {
    if (!sessionId) return;
    let isCancelled = false;

    async function loadHistory() {
      try {
        const res = await fetch(`/api/chat?session_id=${encodeURIComponent(sessionId)}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data && data.history && !isCancelled) {
          const loaded: Message[] = data.history.map((item: any) => {
            let citations: Citation[] = [];
            if (item.citations_json) {
              try {
                citations = JSON.parse(item.citations_json);
              } catch (e) {
                citations = [];
              }
            }
            return {
              sender: item.sender,
              content: item.content,
              citations,
              timestamp: item.timestamp,
            };
          });
          setMessages(loaded);
        }
      } catch (e) {
        console.error("Failed to load history:", e);
      }
    }

    loadHistory();
    return () => {
      isCancelled = true;
    };
  }, [sessionId]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { sender: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    const currentQuery = input.trim();
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: currentQuery }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || errJson.error || "Chat request failed");
      }

      const data = await res.json();
      const agentMsg: Message = {
        sender: "agent",
        content: data.answer,
        citations: data.citations || [],
      };
      setMessages((prev) => [...prev, agentMsg]);
      if (onNewMessage) onNewMessage();
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { sender: "agent", content: `Error: ${err.message || "Unable to fetch grounded response."}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full flex flex-col h-[520px] bg-slate-900/80 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden backdrop-blur-md">
      {/* Header */}
      <div className="p-4 bg-slate-950/70 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center">
            <Bot className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Grounded Text Chat</h3>
            <p className="text-[11px] text-slate-400">LangGraph state machine + Embedded Qdrant citations</p>
          </div>
        </div>
        <span className="text-[10px] font-mono text-slate-400 bg-slate-800/80 px-2 py-1 rounded-md border border-slate-700/50">
          Session: {sessionId.slice(0, 8)}...
        </span>
      </div>

      {/* Message List */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4">
        {messages.length === 0 && !loading && (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500">
            <Sparkles className="w-8 h-8 text-indigo-400/40 mb-2" />
            <p className="text-xs font-medium text-slate-400">No messages in this session yet</p>
            <p className="text-[11px] text-slate-500 mt-1 max-w-xs">
              Ask questions about any indexed knowledge base documents or uploaded files.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <ChatMessageItem key={idx} msg={msg} idx={idx} />
        ))}

        {loading && (
          <div className="flex items-center gap-2 text-xs text-indigo-400 italic bg-indigo-500/10 border border-indigo-500/20 px-3 py-2 rounded-xl w-fit">
            <Bot className="w-4 h-4 animate-bounce shrink-0" />
            <span>Searching vector store & synthesizing grounded answer...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="p-3 bg-slate-950/70 border-t border-slate-800 flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          placeholder="Ask a question about your uploaded documents..."
          disabled={loading}
          className="flex-1 bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="p-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 text-white rounded-xl transition shadow-md shadow-indigo-600/20"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
