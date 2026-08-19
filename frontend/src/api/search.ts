/**
 * api/search.ts
 * Client for index building, background index status, and semantic code search endpoints.
 */
import { API_BASE } from './config'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface IndexSummary {
  files_processed: number
  chunks_indexed: number
  languages: Record<string, number>
}

export interface IndexStatusProgress {
  chunks_processed: number
  total_chunks: number
  percent: number
}

export interface IndexStatusResponse {
  success: boolean
  repo_url: string
  status: 'not_indexed' | 'indexing' | 'ready' | 'failed'
  progress?: IndexStatusProgress | null
  summary?: IndexSummary | null
  error?: string | null
}

export interface BuildIndexSuccess {
  success: true
  repo_url: string
  summary: IndexSummary
}

export interface BuildIndexFailure {
  success: false
  error: string
}

export type BuildIndexResult = BuildIndexSuccess | BuildIndexFailure

export interface SearchResultItem {
  score: number
  chunk_id: string
  file_path: string
  language: string | null
  start_line: number
  end_line: number
  content: string
  line_count: number
  char_count: number
}

export interface SearchSuccess {
  success: true
  query: string
  results: SearchResultItem[]
}

export interface SearchFailure {
  success: false
  error: string
}

export type SearchResult = SearchSuccess | SearchFailure

// ── API calls ─────────────────────────────────────────────────────────────────

/**
 * Fetches the real-time background status of a repository vector index.
 */
export async function getIndexStatus(repoUrl: string): Promise<IndexStatusResponse> {
  const res = await fetch(`${API_BASE}/api/index/status?repo_url=${encodeURIComponent(repoUrl)}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch index status (HTTP ${res.status})`)
  }
  return res.json()
}

/**
 * Triggers vector indexing and polls background progress until completion.
 */
export async function buildSearchIndex(
  repoUrl: string,
  files?: Array<{ path: string; content: string; language?: string | null }>,
  paths?: string[],
  ref?: string,
  onProgress?: (progress: IndexStatusProgress) => void
): Promise<BuildIndexResult> {
  let res: Response

  try {
    res = await fetch(`${API_BASE}/api/index/build`, {
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

  let data: any
  try {
    data = await res.json()
  } catch {
    return {
      success: false,
      error: `Server returned an unexpected response (HTTP ${res.status}).`,
    }
  }

  if (!res.ok || !data.success) {
    const serverError = data?.error || `Server error: HTTP ${res.status}`
    return { success: false, error: serverError }
  }

  // If already ready immediately (e.g. empty or sync mode)
  if (data.status === 'ready' && data.summary) {
    return {
      success: true,
      repo_url: repoUrl,
      summary: data.summary,
    }
  }

  // Poll status endpoint every 1.5 seconds as long as backend reports status === 'indexing'
  // Up to 600 iterations (15 minutes), timeout only if stalled with zero progress for 4 minutes
  const pollIntervalMs = 1500
  const maxPolls = 600 // up to 15 minutes for large repositories on CPU
  let lastProcessedCount = -1
  let staleCount = 0

  for (let i = 0; i < maxPolls; i++) {
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs))

    try {
      const statusRes = await getIndexStatus(repoUrl)

      if (statusRes.progress && onProgress) {
        onProgress(statusRes.progress)
      }

      if (statusRes.status === 'ready' && statusRes.summary) {
        return {
          success: true,
          repo_url: repoUrl,
          summary: statusRes.summary,
        }
      }

      if (statusRes.status === 'failed') {
        return {
          success: false,
          error: statusRes.error || 'Indexing failed in background.',
        }
      }

      // Track progress to detect truly hung/dead tasks
      const currentProcessed = statusRes.progress?.chunks_processed ?? 0
      if (currentProcessed === lastProcessedCount) {
        staleCount++
      } else {
        lastProcessedCount = currentProcessed
        staleCount = 0
      }

      // If no chunks processed for 4 minutes (160 polls), consider hung
      if (staleCount > 160) {
        return {
          success: false,
          error: 'Indexing stalled: No progress received from backend for 4 minutes.',
        }
      }
    } catch {
      // Continue polling on transient network hiccup
    }
  }

  return {
    success: false,
    error: 'Indexing operation exceeded the 15-minute maximum window.',
  }
}

export async function searchCode(
  repoUrl: string,
  query: string,
  topK: number = 5
): Promise<SearchResult> {
  let res: Response

  try {
    res = await fetch(`${API_BASE}/api/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo_url: repoUrl,
        query,
        top_k: topK,
      }),
    })
  } catch {
    return {
      success: false,
      error: 'Could not reach the CodeAtlas search service.',
    }
  }

  let data: unknown
  try {
    data = await res.json()
  } catch {
    return {
      success: false,
      error: `Search service returned an invalid response (HTTP ${res.status}).`,
    }
  }

  if (!res.ok) {
    const serverError =
      typeof data === 'object' && data !== null && 'error' in data
        ? String((data as Record<string, unknown>).error)
        : `Search error: HTTP ${res.status}`
    return { success: false, error: serverError }
  }

  return data as SearchResult
}
