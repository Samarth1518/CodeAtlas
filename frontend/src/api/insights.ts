/**
 * frontend/src/api/insights.ts — Repository Architecture Insights API client.
 *
 * Provides TypeScript interfaces and a typed client function for the
 * POST /api/insights/analyze endpoint (Phase 7).
 */

/** A single tree item (file or directory) */
export interface TreeItem {
  path: string;
  type: "file" | "directory";
  size: number | null;
}

/** A source file with content */
export interface SourceFileInput {
  path: string;
  content: string;
  language?: string | null;
}

/** Repository metadata (from Phase 1) */
export interface RepoMetadataInput {
  name: string;
  owner: string;
  full_name: string;
  primary_language: string | null;
  default_branch: string;
  visibility: string;
}

/** Repository statistics */
export interface InsightStatistics {
  total_files: number;
  total_directories: number;
  total_items: number;
  total_size_bytes: number;
  language_distribution: Record<string, number>;
}

/** Technology detection results */
export interface InsightTechnologies {
  primary_language: string | null;
  detected_languages: string[];
  detected_frameworks: string[];
  manifest_files: string[];
}

/** A parsed dependency manifest */
export interface InsightDependency {
  file: string;
  ecosystem: string;
  parsed: boolean;
  reason?: string;
  data?: {
    name?: string;
    version?: string;
    description?: string;
    dependencies?: string[];
    devDependencies?: string[];
    peerDependencies?: string[];
    packages?: string[];
    gems?: string[];
    module?: string;
    scripts?: string[];
    total_dependencies?: number;
    total_devDependencies?: number;
    total_packages?: number;
    error?: string;
    [key: string]: unknown;
  };
}

/** An important file with its role */
export interface ImportantFile {
  path: string;
  role: string;
  reason: string;
}

/** A directory entry with role */
export interface DirectoryRole {
  name: string;
  role: string;
  file_count: number;
}

/** Directory structure analysis */
export interface InsightDirectoryStructure {
  top_level_directories: string[];
  top_level_files: string[];
  directory_roles: DirectoryRole[];
}

/** Documentation presence */
export interface InsightDocumentation {
  has_readme: boolean;
  documentation_files: string[];
  documentation_file_count: number;
}

/** Full insights response */
export interface RepositoryInsights {
  repo_url: string;
  metadata: {
    name: string | null;
    full_name: string | null;
    owner: string | null;
    primary_language: string | null;
    default_branch: string | null;
    visibility: string | null;
  };
  statistics: InsightStatistics;
  technologies: InsightTechnologies;
  dependencies: InsightDependency[];
  important_files: ImportantFile[];
  directory_structure: InsightDirectoryStructure;
  ci_cd: string[];
  documentation: InsightDocumentation;
  architecture_notes: string[];
}

/** API response types */
export interface InsightsResponse {
  success: true;
  insights: RepositoryInsights;
}

export interface InsightsErrorResponse {
  success: false;
  error: string;
}

import { API_BASE } from './config'

/**
 * Analyzes repository structure and returns architectural insights.
 *
 * @param repoUrl     Full GitHub repository URL
 * @param metadata    Repository metadata from Phase 1 analysis
 * @param tree        File/directory tree from Phase 2
 * @param sourceFiles Optional source files with content from Phase 3
 */
export async function analyzeInsights(
  repoUrl: string,
  metadata: RepoMetadataInput,
  tree: TreeItem[],
  sourceFiles?: SourceFileInput[]
): Promise<InsightsResponse> {
  const response = await fetch(`${API_BASE}/api/insights/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo_url: repoUrl,
      metadata,
      tree,
      source_files: sourceFiles ?? [],
    }),
  });

  const data: InsightsResponse | InsightsErrorResponse = await response.json();

  if (!data.success) {
    throw new Error((data as InsightsErrorResponse).error || "Insights analysis failed.");
  }

  return data as InsightsResponse;
}
