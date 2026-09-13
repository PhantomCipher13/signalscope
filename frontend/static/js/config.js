/**
 * SignalScope — config.js
 *
 * Production: API_BASE is "" (same-origin). Frontend and backend are both
 * served from https://signalscope-kappa.vercel.app — no Cloudflare tunnel,
 * no localhost, no external backend.
 *
 * Local development: also "" when running `uvicorn app.main:app` locally,
 * because FastAPI serves both the frontend and API from the same process.
 *
 * If you need to point to a different backend for testing, change the value
 * here temporarily (do NOT commit the change with a real URL):
 *   window.SIGNALSCOPE_API_BASE = "http://localhost:8000";
 */
window.SIGNALSCOPE_API_BASE = "";
