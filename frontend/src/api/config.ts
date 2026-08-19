/**
 * frontend/src/api/config.ts — Centralized API configuration.
 *
 * Reads VITE_API_BASE_URL from the environment or falls back to http://localhost:5000.
 * Strips any trailing slash to ensure clean route concatenation.
 */

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ||
  "http://localhost:5000";
