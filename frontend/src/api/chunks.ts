/**
 * api/chunks.ts
 * Client for the source-code chunking & index preparation endpoint.
 */
import { API_BASE } from './config'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CodeChunk {
  chunk_id: string
  repo_url: string
  file_path: string
  language: string | null
  start_line: number
  end_line: number
  content: string
  line_count: number
  char_count: number
}

export interface ChunkSummary {
  language_breakdown: Record<string, number>
  total_lines_chunked: number
  truncated: boolean
}

export interface ChunksSuccess {
  success: true
  repo_url: string
  total_chunks: number
  total_files_processed: number
  chunks: CodeChunk[]
  summary: ChunkSummary
}

export interface ChunksFailure {
  success: false
  error: string
}

export type ChunksResult = ChunksSuccess | ChunksFailure

// ── API call ──────────────────────────────────────────────────────────────────

export async function generateChunks(
  repoUrl: string,
  files?: Array<{ path: string; content: string; language?: string | null }>,
  paths?: string[],
  ref?: string
): Promise<ChunksResult> {
  let res: Response

  try {
    res = await fetch(`${API_BASE}/api/chunks/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo_url: repoUrl,
        ...(files ? { files } : {}),
        ...(paths ? { paths, ref } : {}),
      }),
    })
  } catch {
    return {
      success: false,
      error: 'Could not reach the CodeAtlas server. Make sure the backend is running.',
    }
  }

  let data: unknown
  try {
    data = await res.json()
  } catch {
    return {
      success: false,
      error: `Server returned an unexpected response (HTTP ${res.status}).`,
    }
  }

  if (!res.ok) {
    const serverError =
      typeof data === 'object' && data !== null && 'error' in data
        ? String((data as Record<string, unknown>).error)
        : `Server error: HTTP ${res.status}`
    return { success: false, error: serverError }
  }

  return data as ChunksResult
}
