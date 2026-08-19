/**
 * api/repos.ts
 * Thin fetch wrapper for the repository analysis endpoint.
 */
import { API_BASE } from './config'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface RepoMetadata {
  name: string
  owner: string
  full_name: string
  html_url: string
  default_branch: string
  visibility: string
  primary_language: string | null
}

export interface RepoTreeItem {
  path: string
  type: 'file' | 'directory'
  size: number | null
}

export interface TreeSummary {
  total_files: number
  total_dirs: number
  truncated: boolean
}

export interface AnalyzeSuccess {
  success: true
  repo_url: string
  name: string
  owner: string
  full_name: string
  html_url: string
  default_branch: string
  visibility: string
  primary_language: string | null
  metadata?: RepoMetadata
  tree?: RepoTreeItem[]
  tree_summary?: TreeSummary
}

export interface AnalyzeFailure {
  success: false
  error: string
}

export type AnalyzeResult = AnalyzeSuccess | AnalyzeFailure

// ── API call ──────────────────────────────────────────────────────────────────

export async function analyzeRepo(repoUrl: string): Promise<AnalyzeResult> {
  let res: Response

  try {
    res = await fetch(`${API_BASE}/api/repos/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_url: repoUrl }),
    })
  } catch {
    // Network-level failure (server down, no internet, CORS preflight blocked, etc.)
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

  // Server responded with a non-2xx status
  if (!res.ok) {
    const serverError =
      typeof data === 'object' && data !== null && 'error' in data
        ? String((data as Record<string, unknown>).error)
        : `Server error: HTTP ${res.status}`
    return { success: false, error: serverError }
  }

  return data as AnalyzeResult
}
