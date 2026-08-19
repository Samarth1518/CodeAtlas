/**
 * api/search.ts
 * Client for index building and semantic code search endpoints.
 */
import { API_BASE } from './config'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface IndexSummary {
  files_processed: number
  chunks_indexed: number
  languages: Record<string, number>
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

export async function buildSearchIndex(
  repoUrl: string,
  files?: Array<{ path: string; content: string; language?: string | null }>,
  paths?: string[],
  ref?: string
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

  return data as BuildIndexResult
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

  return data as SearchResult
}
