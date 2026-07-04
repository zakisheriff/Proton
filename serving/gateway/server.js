// Proton 1 API gateway
// -----------------------------------------------------------------------------
// The developer-facing front door for The Atom's Proton 1 API. Sits in front of
// the Python inference server (serving/server.py) and adds the things a real API
// product needs: API-key auth, per-key rate limiting, and usage metering.
//
//   client (OpenAI SDK) --> [this gateway :8080] --> [inference server :8000]
//
// Run:
//   cd serving/gateway && npm install && npm start
// Env:
//   PROTON_UPSTREAM   inference server base url (default http://127.0.0.1:8000)
//   PORT              gateway port (default 8080)
//   PROTON_KEYS       comma-separated allowed API keys (default: one demo key)

import express from "express";

const UPSTREAM = process.env.PROTON_UPSTREAM || "http://127.0.0.1:8000";
const PORT = Number(process.env.PORT || 8080);

// --- API keys -----------------------------------------------------------------
// In production these live in a database with per-customer plans. Here: env list.
const KEYS = new Map(
  (process.env.PROTON_KEYS || "sk-proton-demo-key")
    .split(",")
    .map((k) => k.trim())
    .filter(Boolean)
    .map((k) => [k, { plan: "default", rpm: 60 }])
);

// --- in-memory rate limiter + usage meter -------------------------------------
const buckets = new Map(); // key -> { count, windowStart }
const usage = new Map(); // key -> { requests, promptTokens, completionTokens }

function rateLimit(key, rpm) {
  const now = Date.now();
  const b = buckets.get(key) || { count: 0, windowStart: now };
  if (now - b.windowStart >= 60_000) {
    b.count = 0;
    b.windowStart = now;
  }
  b.count += 1;
  buckets.set(key, b);
  return b.count <= rpm;
}

function meter(key, u) {
  const cur = usage.get(key) || { requests: 0, promptTokens: 0, completionTokens: 0 };
  cur.requests += 1;
  cur.promptTokens += u?.prompt_tokens || 0;
  cur.completionTokens += u?.completion_tokens || 0;
  usage.set(key, cur);
}

// --- auth middleware ----------------------------------------------------------
function auth(req, res, next) {
  const header = req.get("authorization") || "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : null;
  if (!token || !KEYS.has(token)) {
    return res.status(401).json({
      error: { message: "Invalid API key.", type: "invalid_request_error", code: "invalid_api_key" },
    });
  }
  const info = KEYS.get(token);
  if (!rateLimit(token, info.rpm)) {
    return res.status(429).json({
      error: { message: "Rate limit exceeded.", type: "rate_limit_error", code: "rate_limit_exceeded" },
    });
  }
  req.apiKey = token;
  next();
}

const app = express();
app.use(express.json({ limit: "10mb" }));

app.get("/health", (_req, res) => res.json({ status: "ok", service: "proton1-gateway" }));

// usage dashboard for a key
app.get("/v1/usage", auth, (req, res) => {
  res.json(usage.get(req.apiKey) || { requests: 0, promptTokens: 0, completionTokens: 0 });
});

// proxy the OpenAI-compatible endpoints to the inference server
for (const path of ["/v1/chat/completions", "/v1/completions"]) {
  app.post(path, auth, async (req, res) => {
    try {
      const upstream = await fetch(`${UPSTREAM}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req.body),
      });

      // stream passthrough
      if (req.body?.stream) {
        res.setHeader("content-type", "text/event-stream");
        res.setHeader("cache-control", "no-cache");
        for await (const chunk of upstream.body) res.write(chunk);
        meter(req.apiKey, null);
        return res.end();
      }

      const data = await upstream.json();
      meter(req.apiKey, data.usage);
      res.status(upstream.status).json(data);
    } catch (err) {
      res.status(502).json({
        error: { message: `Upstream error: ${err.message}`, type: "api_error" },
      });
    }
  });
}

app.get("/v1/models", auth, async (_req, res) => {
  try {
    const upstream = await fetch(`${UPSTREAM}/v1/models`);
    res.status(upstream.status).json(await upstream.json());
  } catch (err) {
    res.status(502).json({ error: { message: err.message, type: "api_error" } });
  }
});

app.listen(PORT, () => {
  console.log(`Proton 1 gateway on :${PORT} -> upstream ${UPSTREAM}`);
  console.log(`Loaded ${KEYS.size} API key(s).`);
});
