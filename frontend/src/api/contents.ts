/**
 * api/contents.ts
 * Client for the source-file content retrieval endpoint.
 */
import { API_BASE } from './config'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SourceFile {
  path: string
  name: string
  size: number
  language: string | null
  encoding: string
  content: string
  sha: string
  html_url: string
}

export interface SkippedFile {
  path: string
  reason: string
}

export interface FetchSummary {
  fetched: number
  skipped: number
  errors: number
}

export interface ContentsSuccess {
  success: true
  owner: string
  repo: string
  ref: string
  files: SourceFile[]
  skipped: SkippedFile[]
  summary: FetchSummary
}

export interface ContentsFailure {
  success: false
  error: string
}

export type ContentsResult = ContentsSuccess | ContentsFailure

// ── API call ──────────────────────────────────────────────────────────────────

export async function fetchContents(
  repoUrl: string,
  paths: string[],
  ref?: string
): Promise<ContentsResult> {
  let res: Response

  try {
    res = await fetch(`${API_BASE}/api/contents/fetch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_url: repoUrl, paths, ref }),
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

  return data as ContentsResult
}
