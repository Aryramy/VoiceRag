"use client";

import React, { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, FileText, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";

interface DocumentUploaderProps {
  onUploadComplete?: () => void;
}

export function DocumentUploader({ onUploadComplete }: DocumentUploaderProps) {
  const [uploading, setUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setStatusMessage(null);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || errorData.detail || "Upload failed");
      }

      setStatusMessage({ type: "success", text: `Uploaded ${file.name}. Ingestion in progress.` });
      if (onUploadComplete) onUploadComplete();
    } catch (err: any) {
      setStatusMessage({ type: "error", text: err.message || "Failed to upload document" });
    } finally {
      setUploading(false);
    }
  }, [onUploadComplete]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
    },
    maxFiles: 1,
  });

  return (
    <div className="w-full bg-slate-900/60 p-4 rounded-xl border border-slate-800 space-y-3">
      <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
        <FileText className="w-4 h-4 text-indigo-400" />
        <span>Upload RAG Documents</span>
      </h3>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center text-center cursor-pointer transition ${
          isDragActive ? "border-indigo-500 bg-indigo-500/10" : "border-slate-700 hover:border-slate-600 bg-slate-950/40"
        }`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="flex flex-col items-center space-y-2 text-indigo-400">
            <Loader2 className="w-8 h-8 animate-spin" />
            <span className="text-xs">Parsing & Indexing Vectors...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-2 text-slate-400">
            <UploadCloud className="w-8 h-8 text-slate-500" />
            <p className="text-xs">
              {isDragActive ? "Drop the file here" : "Drag & drop PDF, TXT, or MD file here, or click to browse"}
            </p>
            <span className="text-[10px] text-slate-500">Max size 25MB • Local embedded Qdrant indexing</span>
          </div>
        )}
      </div>

      {statusMessage && (
        <div
          className={`flex items-center gap-2 p-2.5 rounded-md text-xs border ${
            statusMessage.type === "success"
              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
              : "bg-rose-500/10 text-rose-400 border-rose-500/30"
          }`}
        >
          {statusMessage.type === "success" ? (
            <CheckCircle2 className="w-4 h-4 shrink-0" />
          ) : (
            <AlertCircle className="w-4 h-4 shrink-0" />
          )}
          <span>{statusMessage.text}</span>
        </div>
      )}
    </div>
  );
}

