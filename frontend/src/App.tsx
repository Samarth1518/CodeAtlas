import React, { useState, useEffect, type FormEvent } from 'react'
import './App.css'
import { analyzeRepo, type AnalyzeSuccess, type RepoTreeItem } from './api/repos'
import { fetchContents, type SourceFile } from './api/contents'
import { generateChunks, type CodeChunk, type ChunkSummary } from './api/chunks'
import { buildSearchIndex, searchCode, type IndexSummary, type SearchResultItem } from './api/search'
import { chatWithRepository, type ChatSource } from './api/chat'
import { analyzeInsights, type RepositoryInsights } from './api/insights'

// ── Logo & Brand Icons ────────────────────────────────────────────────────────
function BrandLogo({ size = 32 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 46"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="brand-logo-svg"
      aria-hidden="true"
    >
      <path
        fill="url(#brand-gradient)"
        d="M25.946 44.938c-.664.845-2.021.375-2.021-.698V33.937a2.26 2.26 0 0 0-2.262-2.262H10.287c-.92 0-1.456-1.04-.92-1.788l7.48-10.471c1.07-1.497 0-3.578-1.842-3.578H1.237c-.92 0-1.456-1.04-.92-1.788L10.013.474c.214-.297.556-.474.92-.474h28.894c.92 0 1.456 1.04.92 1.788l-7.48 10.471c-1.07 1.498 0 3.579 1.842 3.579h11.377c.943 0 1.473 1.088.89 1.83L25.947 44.94z"
      />
      <defs>
        <linearGradient id="brand-gradient" x1="0" y1="0" x2="48" y2="46" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="50%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#a855f7" />
        </linearGradient>
      </defs>
    </svg>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined || isNaN(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function langClass(lang: string | null | undefined): string {
  if (!lang) return ''
  return `lang-${lang.toLowerCase().replace(/[^a-z0-9]/g, '')}`
}

function validateUrl(input: string): string | null {
  const trimmed = input.trim()
  if (!trimmed) return 'Please enter a GitHub repository URL.'
  const pattern = /^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(?:\.git)?\/?$/
  if (!pattern.test(trimmed)) {
    return 'Please enter a valid public GitHub URL (e.g. https://github.com/owner/repo).'
  }
  return null
}

function markdownToHtml(md: string): string {
  let html = md
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_m, lang, code) => {
    const langBadge = lang ? `<span class="ai-code-lang">${lang}</span>` : ''
    return `<div class="ai-code-block">${langBadge}<pre><code>${code.trim()}</code></pre></div>`
  })
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>')
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>')
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>')
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>')
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')
  html = html.replace(/`([^`]+)`/g, '<code class="ai-inline-code">$1</code>')
  html = html.replace(/^\s*[-*] (.+)$/gm, '<li>$1</li>')
  html = html.replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
  html = html.replace(/^\s*\d+\. (.+)$/gm, '<li>$1</li>')
  html = html.replace(/^---$/gm, '<hr />')
  html = html.replace(/^(?!<[huplr]|<pre|<hr)(.+)$/gm, '<p>$1</p>')
  html = html.replace(/\n{3,}/g, '\n\n')
  return html
}

function getPrioritizedFiles(tree: RepoTreeItem[] | undefined, limit: number = 20): string[] {
  if (!tree || tree.length === 0) return []
  const fileItems = tree.filter(item => item.type === 'file')
  if (fileItems.length <= limit) return fileItems.map(i => i.path)

  const manifestBasenames = new Set(['package.json', 'requirements.txt', 'setup.py', 'pyproject.toml', 'pipfile', 'cargo.toml', 'go.mod', 'gemfile', 'composer.json', 'pom.xml', 'build.gradle', 'build.gradle.kts', 'mix.exs'])
  const entryBasenames = new Set(['main.py', 'app.py', 'index.ts', 'index.js', 'index.tsx', 'index.jsx', 'main.ts', 'main.js', 'main.tsx', 'main.jsx', 'app.tsx', 'app.jsx', 'app.vue', 'server.js', 'server.ts', 'lib.rs', 'main.rs', 'main.go', 'readme.md', 'license'])

  function getFileScore(path: string): number {
    const filename = path.split('/').pop()?.toLowerCase() || ''
    if (manifestBasenames.has(filename)) return 100
    if (entryBasenames.has(filename)) return 80
    const depth = path.split('/').length
    if (depth <= 2) return 50 - depth
    return 10 - Math.min(depth, 10)
  }

  const sorted = [...fileItems].sort((a, b) => {
    const scoreA = getFileScore(a.path)
    const scoreB = getFileScore(b.path)
    if (scoreA !== scoreB) return scoreB - scoreA
    return a.path.localeCompare(b.path)
  })
  return sorted.slice(0, limit).map(item => item.path)
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('codeatlas-theme')
    if (saved === 'dark' || saved === 'light') return saved
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('codeatlas-theme', theme)
  }, [theme])

  const toggleTheme = () => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))

  const [url, setUrl]                             = useState('')
  const [error, setError]                         = useState<string | null>(null)
  const [apiError, setApiError]                   = useState<string | null>(null)
  const [isLoading, setIsLoading]                 = useState(false)
  const [result, setResult]                       = useState<AnalyzeSuccess | null>(null)
  const [searchFilter, setSearchFilter]           = useState('')

  const [isFetching, setIsFetching]               = useState(false)
  const [fetchError, setFetchError]               = useState<string | null>(null)
  const [sourceFiles, setSourceFiles]             = useState<SourceFile[]>([])
  const [selectedFile, setSelectedFile]           = useState<SourceFile | null>(null)
  const [fetchSummary, setFetchSummary]           = useState<{ fetched: number; skipped: number; errors: number } | null>(null)

  const [isChunking, setIsChunking]               = useState(false)
  const [chunkError, setChunkError]               = useState<string | null>(null)
  const [chunks, setChunks]                       = useState<CodeChunk[]>([])
  const [chunkSummary, setChunkSummary]           = useState<ChunkSummary | null>(null)
  const [selectedChunk, setSelectedChunk]         = useState<CodeChunk | null>(null)
  const [chunkSearchFilter, setChunkSearchFilter] = useState('')

  const [isBuildingIndex, setIsBuildingIndex]     = useState(false)
  const [indexingProgress, setIndexingProgress]   = useState<{ processed: number; total: number } | null>(null)
  const [indexError, setIndexError]               = useState<string | null>(null)
  const [indexSummary, setIndexSummary]           = useState<IndexSummary | null>(null)
  const [searchQuery, setSearchQuery]             = useState('')
  const [isSearching, setIsSearching]             = useState(false)
  const [searchError, setSearchError]             = useState<string | null>(null)
  const [searchResults, setSearchResults]         = useState<SearchResultItem[] | null>(null)

  const [chatQuestion, setChatQuestion]           = useState('')
  const [isAsking, setIsAsking]                   = useState(false)
  const [chatError, setChatError]                 = useState<string | null>(null)
  const [chatAnswer, setChatAnswer]               = useState<string | null>(null)
  const [chatSources, setChatSources]             = useState<ChatSource[]>([])

  const [isAnalyzingInsights, setIsAnalyzingInsights] = useState(false)
  const [insightsError, setInsightsError]             = useState<string | null>(null)
  const [insights, setInsights]                       = useState<RepositoryInsights | null>(null)

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    setUrl(e.target.value)
    if (error)        setError(null)
    if (apiError)     setApiError(null)
    if (result)       setResult(null)
    if (sourceFiles.length) {
      setSourceFiles([])
      setSelectedFile(null)
      setFetchSummary(null)
      setChunks([])
      setSelectedChunk(null)
      setChunkSummary(null)
      setIndexSummary(null)
      setSearchResults(null)
      setChatAnswer(null)
      setChatSources([])
      setChatError(null)
      setInsights(null)
      setInsightsError(null)
    }
  }

  function handleFillSample(sampleUrl: string) {
    setUrl(sampleUrl)
    if (error) setError(null)
    if (apiError) setApiError(null)
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const validationError = validateUrl(url)
    if (validationError) { setError(validationError); return }

    setError(null)
    setApiError(null)
    setResult(null)
    setSearchFilter('')
    setSourceFiles([])
    setSelectedFile(null)
    setFetchSummary(null)
    setFetchError(null)
    setChunks([])
    setSelectedChunk(null)
    setChunkSummary(null)
    setChunkError(null)
    setIndexSummary(null)
    setSearchResults(null)
    setSearchError(null)
    setIsLoading(true)

    try {
      const data = await analyzeRepo(url.trim())
      if (data.success) setResult(data)
      else setApiError(data.error)
    } finally {
      setIsLoading(false)
    }
  }

  async function handleAnalyzeInsights() {
    if (!result) return

    setIsAnalyzingInsights(true)
    setInsightsError(null)
    setInsights(null)

    try {
      const metadataInput = {
        name: result.name || '',
        owner: result.owner || '',
        full_name: result.full_name || '',
        primary_language: result.primary_language || null,
        default_branch: result.default_branch || 'main',
        visibility: result.visibility || 'public',
      }
      const sourceFilesInput = sourceFiles.map(f => ({
        path: f.path,
        content: f.content,
        language: f.language,
      }))
      const data = await analyzeInsights(
        result.repo_url,
        metadataInput,
        result.tree || [],
        sourceFilesInput,
      )
      setInsights(data.insights)
    } catch (err: unknown) {
      setInsightsError(err instanceof Error ? err.message : 'Insights analysis failed.')
    } finally {
      setIsAnalyzingInsights(false)
    }
  }

  async function handleAskQuestion(e: FormEvent) {
    e.preventDefault()
    if (!result || !chatQuestion.trim()) return

    setIsAsking(true)
    setChatError(null)
    setChatAnswer(null)
    setChatSources([])

    try {
      const data = await chatWithRepository(result.repo_url, chatQuestion.trim(), 6)
      setChatAnswer(data.answer)
      setChatSources(data.sources)
    } catch (err: unknown) {
      setChatError(err instanceof Error ? err.message : 'AI request failed.')
    } finally {
      setIsAsking(false)
    }
  }

  async function handleFetchContents() {
    if (!result?.tree) return

    const eligible = getPrioritizedFiles(result.tree, 20)
    if (eligible.length === 0) return

    setIsFetching(true)
    setFetchError(null)
    setSourceFiles([])
    setSelectedFile(null)
    setFetchSummary(null)
    setChunks([])
    setSelectedChunk(null)
    setChunkSummary(null)

    try {
      const data = await fetchContents(result.repo_url, eligible, result.default_branch)
      if (data.success) {
        setSourceFiles(data.files)
        setFetchSummary(data.summary)
        if (data.files.length > 0) setSelectedFile(data.files[0])
      } else {
        setFetchError(data.error)
      }
    } finally {
      setIsFetching(false)
    }
  }

  async function handleGenerateChunks() {
    if (!result) return

    setIsChunking(true)
    setChunkError(null)
    setChunks([])
    setSelectedChunk(null)
    setChunkSummary(null)

    try {
      let data
      if (sourceFiles.length > 0) {
        data = await generateChunks(result.repo_url, sourceFiles.map(f => ({
          path: f.path,
          content: f.content,
          language: f.language,
        })))
      } else {
        const eligible = getPrioritizedFiles(result.tree, 20)
        data = await generateChunks(result.repo_url, undefined, eligible, result.default_branch)
      }

      if (data.success) {
        setChunks(data.chunks)
        setChunkSummary(data.summary)
        if (data.chunks.length > 0) {
          setSelectedChunk(data.chunks[0])
        }
      } else {
        setChunkError(data.error)
      }
    } finally {
      setIsChunking(false)
    }
  }

  async function handleBuildIndex() {
    if (!result) return

    setIsBuildingIndex(true)
    setIndexingProgress(null)
    setIndexError(null)
    setIndexSummary(null)
    setSearchResults(null)

    const onProgress = (p: { chunks_processed: number; total_chunks: number }) => {
      setIndexingProgress({ processed: p.chunks_processed, total: p.total_chunks })
    }

    try {
      let data
      if (sourceFiles.length > 0) {
        data = await buildSearchIndex(
          result.repo_url,
          sourceFiles.map(f => ({
            path: f.path,
            content: f.content,
            language: f.language,
          })),
          undefined,
          undefined,
          onProgress
        )
      } else {
        const eligible = getPrioritizedFiles(result.tree, 20)
        data = await buildSearchIndex(
          result.repo_url,
          undefined,
          eligible,
          result.default_branch,
          onProgress
        )
      }

      if (data.success) {
        setIndexSummary(data.summary)
      } else {
        setIndexError(data.error)
      }
    } finally {
      setIsBuildingIndex(false)
      setIndexingProgress(null)
    }
  }

  async function handleSearch(e: FormEvent) {
    e.preventDefault()
    if (!result || !searchQuery.trim()) return

    setIsSearching(true)
    setSearchError(null)

    try {
      const data = await searchCode(result.repo_url, searchQuery.trim(), 5)
      if (data.success) {
        setSearchResults(data.results)
      } else {
        setSearchError(data.error)
      }
    } finally {
      setIsSearching(false)
    }
  }

  const isDisabled = !url.trim() || isLoading
  const filteredTree = result?.tree?.filter(item =>
    item.path.toLowerCase().includes(searchFilter.toLowerCase())
  ) ?? []

  const filteredChunks = chunks.filter(c =>
    c.file_path.toLowerCase().includes(chunkSearchFilter.toLowerCase()) ||
    c.content.toLowerCase().includes(chunkSearchFilter.toLowerCase())
  )

  return (
    <div className="codeatlas-app">
      {/* ── Futuristic Navbar ── */}
      <header className="navbar">
        <div className="navbar-container">
          <div className="navbar-brand">
            <BrandLogo size={32} />
            <div className="brand-text-group">
              <span className="brand-name">CodeAtlas</span>
              <span className="brand-badge">Intelligence</span>
            </div>
          </div>

          <div className="navbar-tagline-pill">
            <span className="tagline-dot" aria-hidden="true" />
            <span>Map Your Code. Master Your System.</span>
          </div>

          <div className="navbar-actions">
            <a
              href="https://github.com/Samarth1518/CodeAtlas"
              target="_blank"
              rel="noopener noreferrer"
              className="navbar-link-btn"
              title="View CodeAtlas repository on GitHub"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              <span>GitHub</span>
            </a>

            <button
              type="button"
              className="theme-toggle-btn"
              onClick={toggleTheme}
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
              aria-label={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
              )}
              <span className="theme-label">{theme === 'dark' ? 'Light' : 'Dark'}</span>
            </button>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* ── Hero & Search Bar ── */}
        <section className="hero-section" aria-label="CodeAtlas hero">
          <div className="hero-badge">
            <span className="badge-sparkle">✦</span>
            <span>AI-POWERED CODEBASE INTELLIGENCE</span>
          </div>

          <h1 className="hero-main-title">
            Map Your Code. <span className="title-gradient">Master Your System.</span>
          </h1>

          <p className="hero-description">
            Transform any public GitHub repository into an interactive, AI-grounded architectural map.
            Discover code topologies, explore semantic search, and query your system with zero hallucination.
          </p>

          {/* ── Input Card ── */}
          <div className="search-input-card">
            <form onSubmit={handleSubmit} noValidate className="analyze-form">
              <div className="input-field-group">
                <div className="url-input-container">
                  <span className="url-prefix-icon" aria-hidden="true">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
                      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
                    </svg>
                  </span>
                  <input
                    id="repo-url"
                    type="url"
                    className={`main-url-input${error ? ' input-error' : ''}`}
                    placeholder="https://github.com/owner/repository"
                    value={url}
                    onChange={handleChange}
                    autoComplete="off"
                    spellCheck={false}
                    disabled={isLoading}
                    aria-describedby={error ? 'url-error' : apiError ? 'api-error' : result ? 'url-result' : undefined}
                    aria-invalid={!!(error || apiError)}
                  />
                  {url && !isLoading && (
                    <button
                      type="button"
                      className="clear-input-btn"
                      onClick={() => setUrl('')}
                      aria-label="Clear input"
                    >
                      ✕
                    </button>
                  )}
                </div>

                <button
                  id="analyze-btn"
                  type="submit"
                  className={`btn-hero-analyze${isLoading ? ' btn-loading' : ''}`}
                  disabled={isDisabled}
                  aria-busy={isLoading}
                >
                  {isLoading ? (
                    <>
                      <span className="spinner" aria-hidden="true" />
                      <span>Mapping Codebase…</span>
                    </>
                  ) : (
                    <>
                      <span>Analyze Repository</span>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12" />
                        <polyline points="12 5 19 12 12 19" />
                      </svg>
                    </>
                  )}
                </button>
              </div>

              {error && (
                <p id="url-error" className="form-error-msg" role="alert">
                  <span aria-hidden="true">⚠️</span>{error}
                </p>
              )}
              {apiError && (
                <p id="api-error" className="form-error-msg form-error-msg--api" role="alert">
                  <span aria-hidden="true">🔴</span>{apiError}
                </p>
              )}
            </form>

            {/* Quick sample repository pills */}
            <div className="sample-pills-row">
              <span className="sample-label">Try popular repos:</span>
              <div className="sample-pills">
                {[
                  { label: 'CodeAtlas', repo: 'https://github.com/Samarth1518/CodeAtlas' },
                  { label: 'Flask', repo: 'https://github.com/pallets/flask' },
                  { label: 'Express', repo: 'https://github.com/expressjs/express' },
                  { label: 'Hello-World', repo: 'https://github.com/octocat/Hello-World' },
                ].map((s) => (
                  <button
                    key={s.label}
                    type="button"
                    className="sample-pill"
                    onClick={() => handleFillSample(s.repo)}
                    disabled={isLoading}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ── Homepage Atlas Map Centerpiece & Features (when !result) ── */}
          {!result && (
            <div className="homepage-atlas-container" aria-label="CodeAtlas system topology and capabilities">
              {/* ── Visual Map Centerpiece ── */}
              <div className="atlas-map-wrapper">
                <div className="atlas-map-header">
                  <div className="atlas-map-title-row">
                    <span className="atlas-map-badge">✦ LIVE SYSTEM TOPOLOGY</span>
                    <h2 className="atlas-map-title">The CodeAtlas Architectural Map</h2>
                  </div>
                  <p className="atlas-map-subtitle">
                    An intelligent multi-dimensional map that indexes codebases, establishes semantic relationships, and grounds AI responses directly in your repository.
                  </p>
                </div>

                <div className="atlas-canvas-container">
                  {/* Radar & Orbital Rings */}
                  <div className="atlas-radar-rings" aria-hidden="true">
                    <div className="radar-ring radar-ring-1" />
                    <div className="radar-ring radar-ring-2" />
                    <div className="radar-ring radar-ring-3" />
                    <div className="radar-sweep" />
                  </div>

                  {/* Central Intelligence Hub */}
                  <div className="atlas-core-node">
                    <div className="core-glow-pulse" />
                    <div className="core-icon-container">
                      <BrandLogo size={40} />
                    </div>
                    <span className="core-title">CodeAtlas Core</span>
                    <span className="core-status-pill">● Embedding & RAG Hub</span>
                  </div>

                  {/* Constellation Nodes around Hub */}
                  <div className="atlas-node node-frontend">
                    <div className="node-icon">🌐</div>
                    <div className="node-info">
                      <span className="node-name">Client & Structure</span>
                      <span className="node-desc">Recursive File Tree & AST Chunker</span>
                    </div>
                    <span className="node-tag">Phase 1-4</span>
                  </div>

                  <div className="atlas-node node-vector">
                    <div className="node-icon">⚡</div>
                    <div className="node-info">
                      <span className="node-name">Vector Engine</span>
                      <span className="node-desc">Dense ONNX Embeddings & Search Index</span>
                    </div>
                    <span className="node-tag">Phase 5</span>
                  </div>

                  <div className="atlas-node node-insights">
                    <div className="node-icon">🏛️</div>
                    <div className="node-info">
                      <span className="node-name">System Insights</span>
                      <span className="node-desc">Frameworks, Dependencies & Roles</span>
                    </div>
                    <span className="node-tag">Phase 7</span>
                  </div>

                  <div className="atlas-node node-assistant">
                    <div className="node-icon">🤖</div>
                    <div className="node-info">
                      <span className="node-name">AI Assistant</span>
                      <span className="node-desc">Grounded Gemini RAG & Citations</span>
                    </div>
                    <span className="node-tag">Phase 6</span>
                  </div>

                  {/* Connecting Data Beams (SVG) */}
                  <svg className="atlas-connecting-svg" aria-hidden="true">
                    <line x1="20%" y1="30%" x2="50%" y2="50%" className="data-beam beam-1" />
                    <line x1="80%" y1="30%" x2="50%" y2="50%" className="data-beam beam-2" />
                    <line x1="20%" y1="70%" x2="50%" y2="50%" className="data-beam beam-3" />
                    <line x1="80%" y1="70%" x2="50%" y2="50%" className="data-beam beam-4" />
                  </svg>
                </div>
              </div>

              {/* ── Four Core Capability Landmarks ── */}
              <div className="capabilities-grid">
                <div className="capability-card">
                  <div className="capability-icon-box icon-search">🔍</div>
                  <div className="capability-content">
                    <h3 className="capability-title">Dense Semantic Code Search</h3>
                    <p className="capability-desc">
                      Search your codebase using natural language concepts (e.g. <em>"Where is authentication handled?"</em>). Ranked matches highlight exact file paths and line ranges.
                    </p>
                    <span className="capability-metric">384-d ONNX Vector Search</span>
                  </div>
                </div>

                <div className="capability-card">
                  <div className="capability-icon-box icon-ai">🤖</div>
                  <div className="capability-content">
                    <h3 className="capability-title">Grounded AI Assistant</h3>
                    <p className="capability-desc">
                      Ask complex architectural questions. CodeAtlas retrieves relevant semantic chunks and generates precise explanations strictly grounded in source code with zero hallucination.
                    </p>
                    <span className="capability-metric">Line-by-Line Citations</span>
                  </div>
                </div>

                <div className="capability-card">
                  <div className="capability-icon-box icon-insights">🏛️</div>
                  <div className="capability-content">
                    <h3 className="capability-title">Architecture & Dependency Insights</h3>
                    <p className="capability-desc">
                      Automatically detect frameworks, libraries, CI/CD pipelines, manifest dependencies (npm, pip, cargo, go), and directory architecture roles in milliseconds.
                    </p>
                    <span className="capability-metric">Multi-Ecosystem Detection</span>
                  </div>
                </div>

                <div className="capability-card">
                  <div className="capability-icon-box icon-tree">📁</div>
                  <div className="capability-content">
                    <h3 className="capability-title">Repository Explorer & Chunker</h3>
                    <p className="capability-desc">
                      Explore safe recursive file trees, inspect source files with language-aware highlighting, and generate deterministic AST code chunks tailored for LLM reasoning.
                    </p>
                    <span className="capability-metric">Structure-Aware AST Splitting</span>
                  </div>
                </div>
              </div>

              {/* ── How It Works 3-Step Process ── */}
              <div className="how-it-works-section">
                <div className="how-header">
                  <span className="how-badge">PROCESS</span>
                  <h2 className="how-title">How CodeAtlas Works</h2>
                </div>

                <div className="how-steps-grid">
                  <div className="how-step-card">
                    <span className="step-number">01</span>
                    <div className="step-content">
                      <h4 className="step-heading">Connect & Ingest</h4>
                      <p className="step-text">
                        Enter any public GitHub repository URL. CodeAtlas parses the repository tree, detects tech manifests, and prioritizes key entry points.
                      </p>
                    </div>
                  </div>

                  <div className="how-step-card">
                    <span className="step-number">02</span>
                    <div className="step-content">
                      <h4 className="step-heading">Chunk & Vectorize</h4>
                      <p className="step-text">
                        Source files are split into syntax-aware semantic chunks and vectorized locally using dense FastEmbed embeddings.
                      </p>
                    </div>
                  </div>

                  <div className="how-step-card">
                    <span className="step-number">03</span>
                    <div className="step-content">
                      <h4 className="step-heading">Navigate & Query</h4>
                      <p className="step-text">
                        Explore your codebase through ranked semantic search, automatic architectural insights, and citation-backed AI conversations.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ── Results ── */}
        {result && (
          <div id="url-result" className="result-panel" role="status" aria-live="polite">

              {/* Metadata header */}
              <div className="result-header">
                <span className="result-icon" aria-hidden="true">✅</span>
                <div className="result-title-group">
                  <span className="result-label">Repository Metadata</span>
                  <a href={result.html_url || result.repo_url} target="_blank"
                    rel="noopener noreferrer" className="result-repo-name">
                    {result.full_name || result.name || result.repo_url} ↗
                  </a>
                </div>
              </div>

              <div className="result-meta-grid">
                <div className="result-meta-item">
                  <span className="meta-label">Owner</span>
                  <span className="meta-value">{result.owner || '—'}</span>
                </div>
                <div className="result-meta-item">
                  <span className="meta-label">Default Branch</span>
                  <span className="meta-value branch-value">🌿 {result.default_branch || 'main'}</span>
                </div>
                <div className="result-meta-item">
                  <span className="meta-label">Primary Language</span>
                  <span className="meta-value language-tag">{result.primary_language || 'Not specified'}</span>
                </div>
                <div className="result-meta-item">
                  <span className="meta-label">Visibility</span>
                  <span className={`meta-value badge-visibility badge-${result.visibility?.toLowerCase() || 'public'}`}>
                    {result.visibility ? result.visibility.toUpperCase() : 'PUBLIC'}
                  </span>
                </div>
              </div>

              {/* ── Quick Navigation Bar ── */}
              <nav className="quick-nav" aria-label="CodeAtlas sections">
                <a href="#tree-section" className="quick-nav-item">📁 Structure</a>
                <a href="#source-section" className="quick-nav-item">💻 Source Files</a>
                <a href="#chunks-section" className="quick-nav-item">🧩 Chunks</a>
                <a href="#search-section" className="quick-nav-item">🔍 Semantic Search</a>
                <a href="#insights-section" className="quick-nav-item">🏛️ Insights</a>
                <a href="#chat-section" className="quick-nav-item">🤖 AI Assistant</a>
              </nav>

              {/* ── File Tree ── */}
              <div id="tree-section" className="tree-container">
                <div className="tree-header">
                  <div className="tree-header-info">
                    <span className="tree-title">📁 Repository Structure</span>
                    <span className="tree-counts">
                      {result.tree_summary ? (
                        <>
                          <span className="tree-count-badge">{result.tree_summary.total_files} files</span>
                          <span className="tree-count-badge">{result.tree_summary.total_dirs} directories</span>
                          {result.tree_summary.truncated && (
                            <span className="tree-count-badge tree-count-badge--warning">Truncated (Large Repo)</span>
                          )}
                        </>
                      ) : (
                        <span>{result.tree?.length ?? 0} items</span>
                      )}
                    </span>
                  </div>
                  {result.tree && result.tree.length > 5 && (
                    <input type="text" className="tree-filter-input"
                      placeholder="Filter files by path…"
                      value={searchFilter} onChange={e => setSearchFilter(e.target.value)}
                    />
                  )}
                </div>

                {filteredTree.length > 0 ? (
                  <div className="tree-list-wrapper">
                    <ul className="tree-list">
                      {filteredTree.map((item, idx) => (
                        <li key={idx} className={`tree-item tree-item--${item.type}`}>
                          <span className="tree-item-icon" aria-hidden="true">
                            {item.type === 'directory' ? '📁' : '📄'}
                          </span>
                          <span className="tree-item-path" title={item.path}>{item.path}</span>
                          <span className={`tree-item-type-badge type-${item.type}`}>{item.type}</span>
                          {item.type === 'file' && (
                            <span className="tree-item-size">{formatBytes(item.size)}</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <div className="tree-empty-state">
                    {searchFilter ? 'No files match your search filter.' : 'This repository is empty.'}
                  </div>
                )}
              </div>

              {/* ── Phase 3: Source Content Retrieval ── */}
              <div id="source-section" className="source-section">
                <div className="source-section-header">
                  <div className="source-section-title-group">
                    <span className="source-section-title">💻 Source Files</span>
                    <span className="source-section-subtitle">
                      Fetch and view source code from this repository
                    </span>
                  </div>
                  <button
                    id="fetch-source-btn"
                    className={`btn-fetch-source${isFetching ? ' btn-loading' : ''}`}
                    onClick={handleFetchContents}
                    disabled={isFetching || !result.tree?.some(i => i.type === 'file')}
                    aria-busy={isFetching}
                  >
                    {isFetching ? (
                      <><span className="spinner" aria-hidden="true" />Loading…</>
                    ) : sourceFiles.length > 0 ? (
                      '↺ Refresh Sources'
                    ) : (
                      '⬇ Fetch Source Files'
                    )}
                  </button>
                </div>

                {fetchError && (
                  <p className="error-msg error-msg--api" role="alert">
                    <span aria-hidden="true">🔴</span>{fetchError}
                  </p>
                )}

                {fetchSummary && (
                  <div className="fetch-summary">
                    <span className="fetch-badge fetch-badge--ok">✓ {fetchSummary.fetched} fetched</span>
                    {fetchSummary.skipped > 0 && (
                      <span className="fetch-badge fetch-badge--skip">{fetchSummary.skipped} skipped</span>
                    )}
                    {fetchSummary.errors > 0 && (
                      <span className="fetch-badge fetch-badge--err">{fetchSummary.errors} errors</span>
                    )}
                  </div>
                )}

                {sourceFiles.length > 0 && (
                  <div className="source-viewer">
                    {/* File list sidebar */}
                    <div className="source-file-list">
                      {sourceFiles.map((file, idx) => (
                        <button
                          key={idx}
                          className={`source-file-btn${selectedFile?.path === file.path ? ' active' : ''}`}
                          onClick={() => setSelectedFile(file)}
                          title={file.path}
                        >
                          <span className="source-file-icon" aria-hidden="true">📄</span>
                          <span className="source-file-name">{file.name}</span>
                          {file.language && (
                            <span className="source-file-lang">{file.language}</span>
                          )}
                        </button>
                      ))}
                    </div>

                    {/* Code display panel */}
                    {selectedFile && (
                      <div className="source-code-panel">
                        <div className="source-code-header">
                          <span className="source-code-path">{selectedFile.path}</span>
                          <div className="source-code-meta">
                            {selectedFile.language && (
                              <span className="source-lang-badge">{selectedFile.language}</span>
                            )}
                            <span className="source-size-badge">{formatBytes(selectedFile.size)}</span>
                            <a href={selectedFile.html_url} target="_blank" rel="noopener noreferrer"
                              className="source-gh-link">View on GitHub ↗</a>
                          </div>
                        </div>
                        <pre className={`source-code-pre ${langClass(selectedFile.language)}`}>
                          <code>{selectedFile.content}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── Phase 4: Code Chunking & Index Preparation ── */}
              <div id="chunks-section" className="chunks-section">
                <div className="chunks-section-header">
                  <div className="chunks-section-title-group">
                    <span className="chunks-section-title">🧩 Code Chunk Index</span>
                    <span className="chunks-section-subtitle">
                      Process source code into semantic chunks prepared for vector embeddings & RAG
                    </span>
                  </div>
                  <button
                    id="generate-chunks-btn"
                    className={`btn-generate-chunks${isChunking ? ' btn-loading' : ''}`}
                    onClick={handleGenerateChunks}
                    disabled={isChunking || (!sourceFiles.length && !result.tree?.some(i => i.type === 'file'))}
                    aria-busy={isChunking}
                  >
                    {isChunking ? (
                      <><span className="spinner" aria-hidden="true" />Generating Chunks…</>
                    ) : chunks.length > 0 ? (
                      '↺ Regenerate Chunks'
                    ) : (
                      '⚡ Generate Chunks'
                    )}
                  </button>
                </div>

                {chunkError && (
                  <p className="error-msg error-msg--api" role="alert">
                    <span aria-hidden="true">🔴</span>{chunkError}
                  </p>
                )}

                {chunkSummary && (
                  <div className="chunks-summary-container">
                    <div className="chunks-stat-cards">
                      <div className="chunk-stat-card">
                        <span className="stat-num">{chunks.length}</span>
                        <span className="stat-label">Total Chunks</span>
                      </div>
                      <div className="chunk-stat-card">
                        <span className="stat-num">{chunkSummary.total_lines_chunked}</span>
                        <span className="stat-label">Lines Processed</span>
                      </div>
                    </div>

                    {Object.keys(chunkSummary.language_breakdown).length > 0 && (
                      <div className="chunk-lang-tags">
                        {Object.entries(chunkSummary.language_breakdown).map(([lang, count]) => (
                          <span key={lang} className="chunk-lang-tag">
                            {lang}: <strong>{count}</strong>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {chunks.length > 0 && (
                  <div className="chunks-viewer">
                    {/* Chunk navigation sidebar */}
                    <div className="chunks-sidebar">
                      {chunks.length > 5 && (
                        <input
                          type="text"
                          className="chunk-filter-input"
                          placeholder="Filter chunks by path or keyword…"
                          value={chunkSearchFilter}
                          onChange={e => setChunkSearchFilter(e.target.value)}
                        />
                      )}
                      <div className="chunks-list">
                        {filteredChunks.map((chunk, idx) => (
                          <button
                            key={chunk.chunk_id || idx}
                            className={`chunk-list-item${selectedChunk?.chunk_id === chunk.chunk_id ? ' active' : ''}`}
                            onClick={() => setSelectedChunk(chunk)}
                          >
                            <div className="chunk-item-top">
                              <span className="chunk-item-path" title={chunk.file_path}>
                                {chunk.file_path.split('/').pop()}
                              </span>
                              <span className="chunk-line-badge">
                                L{chunk.start_line}–L{chunk.end_line}
                              </span>
                            </div>
                            <div className="chunk-item-sub">
                              <span className="chunk-item-full-path">{chunk.file_path}</span>
                              <span className="chunk-item-lines">{chunk.line_count} lines</span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Chunk preview panel */}
                    {selectedChunk && (
                      <div className="chunk-preview-panel">
                        <div className="chunk-preview-header">
                          <div className="chunk-preview-meta">
                            <span className="chunk-preview-path">{selectedChunk.file_path}</span>
                            <span className="chunk-preview-lines">
                              Lines {selectedChunk.start_line}–{selectedChunk.end_line} ({selectedChunk.line_count} lines, {selectedChunk.char_count} chars)
                            </span>
                          </div>
                          <div className="chunk-preview-badges">
                            {selectedChunk.language && (
                              <span className="source-lang-badge">{selectedChunk.language}</span>
                            )}
                            <span className="chunk-id-tag" title={selectedChunk.chunk_id}>
                              ID: {selectedChunk.chunk_id.split(':').slice(-1)[0]}
                            </span>
                          </div>
                        </div>
                        <pre className={`chunk-preview-code ${langClass(selectedChunk.language)}`}>
                          <code>{selectedChunk.content}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── Phase 5: Embeddings & Semantic Code Search ── */}
              <div id="search-section" className="search-section">
                <div className="search-section-header">
                  <div className="search-section-title-group">
                    <span className="search-section-title">🔍 Semantic Code Search</span>
                    <span className="search-section-subtitle">
                      Ask questions in natural language and retrieve ranked semantic code matches
                    </span>
                  </div>
                  <button
                    id="build-index-btn"
                    className={`btn-build-index${isBuildingIndex ? ' btn-loading' : ''}`}
                    onClick={handleBuildIndex}
                    disabled={isBuildingIndex || (!sourceFiles.length && !result.tree?.some(i => i.type === 'file'))}
                    aria-busy={isBuildingIndex}
                  >
                    {isBuildingIndex ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        {indexingProgress && indexingProgress.total > 0
                          ? `Indexing (${indexingProgress.processed}/${indexingProgress.total})…`
                          : 'Embedding & Indexing…'}
                      </>
                    ) : indexSummary ? (
                      '✓ Search Index Ready (Rebuild)'
                    ) : (
                      '⚡ Build Search Index'
                    )}
                  </button>
                </div>

                {indexError && (
                  <p className="error-msg error-msg--api" role="alert">
                    <span aria-hidden="true">🔴</span>{indexError}
                  </p>
                )}

                {indexSummary && (
                  <div className="index-summary-bar">
                    <span className="index-summary-badge">
                      ✓ Vector Index: <strong>{indexSummary.chunks_indexed} chunks</strong> across <strong>{indexSummary.files_processed} files</strong>
                    </span>
                  </div>
                )}

                {/* Search query form */}
                <form onSubmit={handleSearch} className="search-form">
                  <div className="search-input-wrapper">
                    <span className="search-icon" aria-hidden="true">🔎</span>
                    <input
                      id="semantic-search-input"
                      type="text"
                      className="search-input"
                      placeholder="Ask a question or search for code concepts (e.g. 'Where is authentication handled?')..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      disabled={isSearching}
                      maxLength={500}
                    />
                    <button
                      id="search-btn"
                      type="submit"
                      className={`btn-search${isSearching ? ' btn-loading' : ''}`}
                      disabled={isSearching || !searchQuery.trim()}
                      aria-busy={isSearching}
                    >
                      {isSearching ? (
                        <><span className="spinner" aria-hidden="true" />Searching…</>
                      ) : (
                        'Search Code'
                      )}
                    </button>
                  </div>
                </form>

                {searchError && (
                  <p className="error-msg error-msg--api" role="alert">
                    <span aria-hidden="true">🔴</span>{searchError}
                  </p>
                )}

                {/* Search Results list */}
                {searchResults !== null && (
                  <div className="search-results-container">
                    <div className="search-results-header">
                      <span className="search-results-title">
                        Ranked Semantic Matches ({searchResults.length})
                      </span>
                    </div>

                    {searchResults.length > 0 ? (
                      <div className="search-results-list">
                        {searchResults.map((item, idx) => (
                          <div key={item.chunk_id || idx} className="search-result-card">
                            <div className="result-card-header">
                              <div className="result-card-title-group">
                                <span className="result-score-badge">
                                  {Math.round(item.score * 100)}% match
                                </span>
                                <span className="result-file-path">{item.file_path}</span>
                              </div>
                              <div className="result-card-meta">
                                {item.language && (
                                  <span className="source-lang-badge">{item.language}</span>
                                )}
                                <span className="chunk-line-badge">
                                  L{item.start_line}–L{item.end_line}
                                </span>
                                <a
                                  href={`${result.html_url || result.repo_url}/blob/${result.default_branch || 'main'}/${item.file_path}#L${item.start_line}-L${item.end_line}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="source-gh-link"
                                >
                                  Open on GitHub ↗
                                </a>
                              </div>
                            </div>
                            <pre className={`result-code-snippet ${langClass(item.language)}`}>
                              <code>{item.content}</code>
                            </pre>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="tree-empty-state">
                        No semantic code matches found for your query. Try rephrasing or asking a broader question.
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* ── Phase 7: Repository Architecture Insights ── */}
              <div id="insights-section" className="insights-section">
                <div className="insights-section-header">
                  <div className="insights-title-group">
                    <span className="insights-title">🏛️ Repository Insights</span>
                    <span className="insights-subtitle">
                      Architecture overview, technology stack, dependencies, and code structure
                    </span>
                  </div>
                  <button
                    id="analyze-insights-btn"
                    className={`btn-insights${isAnalyzingInsights ? ' btn-loading' : ''}`}
                    onClick={handleAnalyzeInsights}
                    disabled={isAnalyzingInsights || !result?.tree?.length}
                    aria-busy={isAnalyzingInsights}
                  >
                    {isAnalyzingInsights ? (
                      <><span className="spinner" aria-hidden="true" />Analyzing…</>
                    ) : insights ? (
                      '↺ Re-analyze'
                    ) : (
                      '🔭 Analyze Architecture'
                    )}
                  </button>
                </div>

                {insightsError && (
                  <p className="error-msg error-msg--api" role="alert">
                    <span aria-hidden="true">🔴</span>{insightsError}
                  </p>
                )}

                {insights && (
                  <div className="insights-panel">

                    {/* ── Statistics row ── */}
                    <div className="insights-stats-row">
                      <div className="insight-stat-card">
                        <span className="insight-stat-num">{insights.statistics.total_files}</span>
                        <span className="insight-stat-label">Files</span>
                      </div>
                      <div className="insight-stat-card">
                        <span className="insight-stat-num">{insights.statistics.total_directories}</span>
                        <span className="insight-stat-label">Directories</span>
                      </div>
                      <div className="insight-stat-card">
                        <span className="insight-stat-num">
                          {insights.statistics.total_size_bytes > 0
                            ? formatBytes(insights.statistics.total_size_bytes)
                            : '—'}
                        </span>
                        <span className="insight-stat-label">Total Size</span>
                      </div>
                      <div className="insight-stat-card">
                        <span className="insight-stat-num">
                          {Object.keys(insights.statistics.language_distribution).length}
                        </span>
                        <span className="insight-stat-label">Languages</span>
                      </div>
                    </div>

                    {/* ── Technology stack ── */}
                    <div className="insights-block">
                      <div className="insights-block-title">⚙️ Technology Stack</div>
                      <div className="insights-block-body">
                        {insights.technologies.detected_frameworks.length > 0 && (
                          <div className="insights-tag-group">
                            <span className="insights-tag-label">Frameworks & Libraries</span>
                            <div className="insights-tags">
                              {insights.technologies.detected_frameworks.map(f => (
                                <span key={f} className="insight-tag insight-tag--framework">{f}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {insights.technologies.detected_languages.length > 0 && (
                          <div className="insights-tag-group">
                            <span className="insights-tag-label">Languages Detected</span>
                            <div className="insights-tags">
                              {insights.technologies.detected_languages.map(l => (
                                <span key={l} className="insight-tag insight-tag--lang">{l}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {Object.keys(insights.statistics.language_distribution).length > 0 && (
                          <div className="insights-tag-group">
                            <span className="insights-tag-label">By File Count</span>
                            <div className="insights-tags">
                              {Object.entries(insights.statistics.language_distribution)
                                .sort(([, a], [, b]) => b - a)
                                .slice(0, 10)
                                .map(([lang, count]) => (
                                  <span key={lang} className="insight-tag insight-tag--count">
                                    {lang}: <strong>{count}</strong>
                                  </span>
                                ))}
                            </div>
                          </div>
                        )}
                        {insights.ci_cd.length > 0 && (
                          <div className="insights-tag-group">
                            <span className="insights-tag-label">CI/CD & DevOps</span>
                            <div className="insights-tags">
                              {insights.ci_cd.map(ci => (
                                <span key={ci} className="insight-tag insight-tag--ci">{ci}</span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* ── Architecture notes ── */}
                    {insights.architecture_notes.length > 0 && (
                      <div className="insights-block">
                        <div className="insights-block-title">🔍 Architecture Observations</div>
                        <ul className="insights-notes-list">
                          {insights.architecture_notes.map((note, i) => (
                            <li key={i} className="insights-note-item">
                              <span className="insights-note-bullet" aria-hidden="true">→</span>
                              {note}
                            </li>
                          ))}
                        </ul>
                        <p className="insights-grounding-notice">
                          These observations are derived directly from the retrieved repository structure and files — no information is invented.
                        </p>
                      </div>
                    )}

                    {/* ── Dependencies ── */}
                    {insights.dependencies.length > 0 && (
                      <div className="insights-block">
                        <div className="insights-block-title">📦 Dependencies</div>
                        <div className="insights-dep-list">
                          {insights.dependencies.map((dep, i) => (
                            <div key={i} className="insights-dep-card">
                              <div className="insights-dep-header">
                                <span className="insights-dep-file">{dep.file}</span>
                                <span className={`insights-dep-badge ${dep.parsed ? 'badge--ok' : 'badge--skip'}`}>
                                  {dep.ecosystem}
                                </span>
                              </div>
                              {dep.parsed && dep.data && !dep.data.error && (
                                <div className="insights-dep-body">
                                  {dep.data.name && (
                                    <span className="insights-dep-name">
                                      {dep.data.name}{dep.data.version ? ` v${dep.data.version}` : ''}
                                    </span>
                                  )}
                                  {dep.data.packages && dep.data.packages.length > 0 && (
                                    <div className="insights-dep-pkgs">
                                      {dep.data.packages.slice(0, 15).map((pkg: string) => (
                                        <span key={pkg} className="insight-tag insight-tag--dep">{pkg}</span>
                                      ))}
                                      {(dep.data.total_packages ?? 0) > 15 && (
                                        <span className="insight-tag insight-tag--more">
                                          +{(dep.data.total_packages ?? 0) - 15} more
                                        </span>
                                      )}
                                    </div>
                                  )}
                                  {dep.data.dependencies && dep.data.dependencies.length > 0 && (
                                    <div className="insights-dep-pkgs">
                                      <span className="insights-tag-label">Dependencies ({dep.data.total_dependencies ?? dep.data.dependencies.length})</span>
                                      <div>
                                        {dep.data.dependencies.slice(0, 15).map((pkg: string) => (
                                          <span key={pkg} className="insight-tag insight-tag--dep">{pkg}</span>
                                        ))}
                                        {(dep.data.total_dependencies ?? 0) > 15 && (
                                          <span className="insight-tag insight-tag--more">
                                            +{(dep.data.total_dependencies ?? 0) - 15} more
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                  {dep.data.devDependencies && dep.data.devDependencies.length > 0 && (
                                    <div className="insights-dep-pkgs">
                                      <span className="insights-tag-label">Dev ({dep.data.total_devDependencies ?? dep.data.devDependencies.length})</span>
                                      <div>
                                        {dep.data.devDependencies.slice(0, 10).map((pkg: string) => (
                                          <span key={pkg} className="insight-tag insight-tag--dev">{pkg}</span>
                                        ))}
                                        {(dep.data.total_devDependencies ?? 0) > 10 && (
                                          <span className="insight-tag insight-tag--more">
                                            +{(dep.data.total_devDependencies ?? 0) - 10} more
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              {!dep.parsed && (
                                <span className="insights-dep-reason">{dep.reason}</span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* ── Important files ── */}
                    {insights.important_files.length > 0 && (
                      <div className="insights-block">
                        <div className="insights-block-title">⭐ Important Files</div>
                        <div className="insights-files-grid">
                          {insights.important_files.map((f, i) => (
                            <div key={i} className="insights-file-card">
                              <div className="insights-file-top">
                                <span className="insights-file-path">{f.path}</span>
                                <span className="insights-file-role">{f.role}</span>
                              </div>
                              <span className="insights-file-reason">{f.reason}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* ── Directory structure ── */}
                    {insights.directory_structure.directory_roles.length > 0 && (
                      <div className="insights-block">
                        <div className="insights-block-title">📂 Directory Structure</div>
                        <div className="insights-dirs-grid">
                          {insights.directory_structure.directory_roles
                            .filter(d => d.role !== 'Unknown')
                            .slice(0, 16)
                            .map((d, i) => (
                              <div key={i} className="insights-dir-card">
                                <span className="insights-dir-name">/{d.name}</span>
                                <span className="insights-dir-role">{d.role}</span>
                                {d.file_count > 0 && (
                                  <span className="insights-dir-count">{d.file_count} files</span>
                                )}
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                  </div>
                )}
              </div>

              {/* ── Phase 6: AI Codebase Assistant ── */}
              <div id="chat-section" className="ai-chat-section">
                <div className="ai-chat-header">
                  <div className="ai-chat-title-group">
                    <span className="ai-chat-title">🤖 AI Codebase Assistant</span>
                    <span className="ai-chat-subtitle">
                      Ask questions about this repository and get answers grounded in its source code.
                    </span>
                  </div>
                </div>

                {/* Example question chips */}
                <div className="ai-example-chips">
                  {[
                    'How does this application work?',
                    'Where is authentication handled?',
                    'Explain the backend architecture.',
                    'How does data flow through the application?',
                    'What are the most important files?',
                    'Where are API endpoints defined?',
                  ].map((q) => (
                    <button
                      key={q}
                      className="ai-example-chip"
                      onClick={() => setChatQuestion(q)}
                      disabled={isAsking}
                      type="button"
                    >
                      {q}
                    </button>
                  ))}
                </div>

                {/* Question form */}
                <form onSubmit={handleAskQuestion} className="ai-question-form">
                  <div className="ai-input-wrapper">
                    <span className="ai-input-icon" aria-hidden="true">💬</span>
                    <textarea
                      id="ai-question-input"
                      className="ai-question-textarea"
                      placeholder="Ask anything about this codebase… (e.g. 'How does authentication work?')"
                      value={chatQuestion}
                      onChange={e => setChatQuestion(e.target.value)}
                      disabled={isAsking}
                      maxLength={500}
                      rows={2}
                    />
                  </div>
                  <div className="ai-form-footer">
                    <span className="ai-char-count">{chatQuestion.length}/500</span>
                    <button
                      id="ask-codeatlas-btn"
                      type="submit"
                      className={`btn-ask${isAsking ? ' btn-loading' : ''}`}
                      disabled={isAsking || !chatQuestion.trim() || !indexSummary}
                      aria-busy={isAsking}
                    >
                      {isAsking ? (
                        <><span className="spinner" aria-hidden="true" />Thinking…</>
                      ) : (
                        '✨ Ask CodeAtlas'
                      )}
                    </button>
                  </div>
                  {!indexSummary && (
                    <p className="ai-index-notice">
                      ⚠️ Build the Search Index above before asking questions.
                    </p>
                  )}
                </form>

                {chatError && (
                  <div className="ai-error" role="alert">
                    <span aria-hidden="true">🔴</span>
                    <div className="ai-error-content">
                      <div className="ai-error-text">{chatError}</div>
                      {chatError.includes('LLM_API_KEY') && (
                        <div className="ai-error-hint">
                          💡 <strong>How to enable:</strong> Add <code>LLM_API_KEY=your_key</code> in <code>backend/.env</code> and restart the backend server.
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* AI Answer panel */}
                {chatAnswer && (
                  <div className="ai-answer-panel">
                    <div className="ai-answer-header">
                      <span className="ai-answer-label">🧠 AI Analysis</span>
                      <span className="ai-answer-badge">Grounded in repository code</span>
                    </div>
                    <div
                      className="ai-answer-body"
                      dangerouslySetInnerHTML={{ __html: markdownToHtml(chatAnswer) }}
                    />

                    {/* Source citations */}
                    {chatSources.length > 0 && (
                      <div className="ai-sources">
                        <div className="ai-sources-title">📎 Sources</div>
                        <div className="ai-sources-list">
                          {chatSources.map((src, idx) => (
                            <div key={src.chunk_id || idx} className="ai-source-card">
                              <div className="ai-source-top">
                                <span className="ai-source-path">{src.file_path}</span>
                                <span className="ai-source-score">
                                  {Math.round(src.score * 100)}% match
                                </span>
                              </div>
                              <div className="ai-source-meta">
                                {src.language && (
                                  <span className="source-lang-badge">{src.language}</span>
                                )}
                                <span className="chunk-line-badge">
                                  Lines {src.start_line}–{src.end_line}
                                </span>
                                <a
                                  href={`${result.html_url || result.repo_url}/blob/${result.default_branch || 'main'}/${src.file_path}#L${src.start_line}-L${src.end_line}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="source-gh-link"
                                >
                                  Open on GitHub ↗
                                </a>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

            </div>
          )}

        {/* ── Footer ── */}
        <footer className="app-footer">
          <div className="footer-content">
            <div className="footer-brand">
              <BrandLogo size={24} />
              <span className="footer-brand-name">CodeAtlas</span>
            </div>
            <p className="footer-tagline">Map Your Code. Master Your System.</p>
            <div className="footer-links">
              <a
                href="https://github.com/Samarth1518/CodeAtlas"
                target="_blank"
                rel="noopener noreferrer"
                className="footer-link"
              >
                GitHub Repository
              </a>
              <span className="footer-divider">•</span>
              <span className="footer-copyright">© {new Date().getFullYear()} CodeAtlas</span>
            </div>
          </div>
        </footer>
      </main>
    </div>
  )
}

