/**
 * frontend/src/api/chat.ts — AI Codebase Assistant API client.
 *
 * Provides TypeScript interfaces and a typed client function for the
 * POST /api/chat endpoint (Phase 6 RAG pipeline).
 */

/** Request payload for POST /api/chat */
export interface ChatRequest {
  repo_url: string;
  question: string;
  top_k?: number;
}

/** A single source citation returned from the RAG pipeline */
export interface ChatSource {
  chunk_id: string;
  file_path: string;
  language: string | null;
  start_line: number;
  end_line: number;
  score: number;
}

/** Response from POST /api/chat on success */
export interface ChatResponse {
  success: true;
  answer: string;
  sources: ChatSource[];
}

/** Response from POST /api/chat on error */
export interface ChatErrorResponse {
  success: false;
  error: string;
}

import { API_BASE } from './config'

/**
 * Sends a natural-language question about a repository and receives a
 * RAG-grounded answer with source citations.
 *
 * @param repoUrl   Full GitHub repository URL
 * @param question  Developer's natural-language question (max 500 chars)
 * @param topK      Number of code chunks to retrieve (max 10, default 6)
 */
export async function chatWithRepository(
  repoUrl: string,
  question: string,
  topK: number = 6
): Promise<ChatResponse> {
  const payload: ChatRequest = {
    repo_url: repoUrl,
    question,
    top_k: Math.min(Math.max(1, topK), 10),
  };

  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data: ChatResponse | ChatErrorResponse = await response.json();

  if (!data.success) {
    throw new Error((data as ChatErrorResponse).error || "AI chat request failed.");
  }

  return data as ChatResponse;
}
