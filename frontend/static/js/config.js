/**
 * SignalScope — config.js
 * Injected by the build/deployment system to configure the API base URL.
 *
 * In Vercel production: vercel.json rewrites /analyze, /health, /feedback
 * to the backend automatically, so API_BASE stays "" (same-origin).
 *
 * If you are running the frontend standalone against a separate backend,
 * set SIGNALSCOPE_API_BASE to the backend URL here:
 *   window.SIGNALSCOPE_API_BASE = "https://your-backend.example.com";
 *
 * Leave empty ("") when the frontend is served by FastAPI directly (local dev)
 * or when Vercel rewrites handle the proxying.
 */
window.SIGNALSCOPE_API_BASE = "";
