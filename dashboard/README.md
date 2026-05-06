# EPS Dashboard

Public research dashboard for `explore-persona-space`. Canonical store for
**claims, experiments, runs, todos, edges, agent tasks**, and **chat sessions**
with a spawned Claude Code agent.

Stack: Next.js 15 · React 19 · Drizzle · Supabase (Postgres + auth + realtime
+ storage) · React Flow · TipTap · Tailwind · Vercel.

This `dashboard/` folder is a self-contained Next.js project inside the
`explore-persona-space` monorepo. It deploys to Vercel with `dashboard/` as
the root directory.

## Access model

- **Public read** (no login): `/graph`, `/claim/[id]`, `/experiment/[id]`,
  `/run/[id]`, `/timeline/today`, `/timeline/week`, `/live`, `/todos`.
- **Auth required** (only the project owner, signed in via Supabase magic
  link to your email): chat rail, write endpoints (TipTap save, edge
  add/remove, todo CRUD, "create claim").
- Enforced two ways: middleware redirects protected routes to `/login`;
  Supabase RLS policies (`db/policies.sql`) make `anon` SELECT-only.

## Milestone status (MVP-B, ~7-8 days)

- [x] **M1** Scaffold + DB schema + base layout + RLS policies
- [ ] **M2** Backfill `clean-results`-labeled GH issues into `claim` + edges
- [ ] **M3** Graph view (React Flow)
- [ ] **M4** Detail pages (`/claim/[id]`, `/experiment/[id]`, `/run/[id]`)
- [ ] **M5** Timeline (`/today`, `/week`)
- [ ] **M6** TipTap inline editing + edge add/remove + Todos CRUD
- [ ] **M7** Chat rail backed by spawned Claude Code agent (Cloudflare Tunnel
  → Python sidecar on the VM)

## One-time setup

We deploy straight to Vercel — no local dev workflow. The Next.js code
runs on the VM only for the build step (or directly on Vercel from a git
push). All viewing happens through the public Vercel URL.

### 1. Supabase project

1. https://supabase.com → **New project** (any region, free tier is fine).
   Save the database password somewhere — you'll need it for `DATABASE_URL`.
2. Project Settings → **API** → copy:
   - `Project URL` → `NEXT_PUBLIC_SUPABASE_URL`
   - `Publishable key` (`sb_publishable_...`) → `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`
   - `Secret key` (`sb_secret_...`, click Reveal) → `SUPABASE_SECRET_KEY`
3. Project → **Connect** button at top → **App Frameworks** → **Drizzle** /
   **Postgres** → **Transaction pooler** (port 6543) → copy → paste as
   `DATABASE_URL`. Replace `[YOUR-PASSWORD]` placeholder.

### 2. Auth: magic link (no setup needed)

Supabase's Email provider is on by default. The login page uses
`signInWithOtp({ email })` — type your email, get a magic link, click,
you're in. No OAuth provider setup required.

After Vercel deploy (Stage 4), add the Vercel URL to Supabase Auth → URL
Configuration → **Redirect URLs**, otherwise the link will refuse to
redirect back.

### 3. Push schema + RLS policies

On the VM, with `dashboard/.env.local` filled in:

```bash
cd dashboard
cp .env.example .env.local         # then fill in values from steps 1 & 2
pnpm install
pnpm db:push                       # creates all tables in Supabase
pnpm db:init-rls                   # applies db/policies.sql (public-read, auth-write)
```

### 4. Vercel deploy

1. Vercel → **Add New** → **Project** → import
   `superkaiba/explore-persona-space`.
2. **Root directory**: `dashboard`.
3. **Environment variables** — paste the same values from `.env.local`,
   plus `NEXT_PUBLIC_SITE_URL=https://<your-vercel-url>`.
4. Deploy.
5. Once deployed, update Supabase Auth → URL Configuration → **Site URL**
   to match the Vercel URL, and add it to the **Redirect URLs** list.
6. Update the GitHub OAuth app's Homepage URL to the Vercel URL too.

### 5. Cloudflare Tunnel (deferred to milestone 7)

When the chat agent ships, expose the VM-hosted Python sidecar at a stable
URL the Vercel app can call:

```bash
cloudflared tunnel login
cloudflared tunnel create eps-chat
# create ~/.cloudflared/config.yml: ingress -> http://localhost:<sidecar-port>
cloudflared tunnel route dns eps-chat chat.YOUR-DOMAIN.com
sudo cloudflared service install
```

Then set `NEXT_PUBLIC_SIDECAR_URL=https://chat.YOUR-DOMAIN.com` in Vercel.

## Layout

```
app/                          # Next.js App Router
  api/auth/callback/route.ts  Supabase OAuth code→session exchange
  api/chat/                   Chat sidecar proxy [M7] — auth-required
  graph/                      Claim graph (React Flow) [M3]
  live/                       Agent kanban [M7+]
  todos/                      Todo board [M6]
  timeline/{today,week}/      Day/week summaries [M5]
  claim/[id]/                 Claim detail [M4]
  experiment/[id]/            Experiment detail [M4]
  run/[id]/                   Run detail [M4]
  login/                      GitHub OAuth (only the project owner)
components/
  nav/Sidebar.tsx             Left nav
  chat/ChatRail.tsx           Right rail; backend in M7
  graph/                      React Flow components [M3]
  editor/                     TipTap editor [M6]
db/
  schema.ts                   Drizzle schema (entities + edges + chat)
  client.ts                   Lazy postgres-js + drizzle, pooler-friendly
  policies.sql                RLS: public-read on entities, auth-only on chat
  migrations/                 generated by drizzle-kit
lib/supabase/
  client.ts                   Browser client
  server.ts                   SSR client (cookies-aware)
  middleware.ts               Cookie refresh + auth gate for protected routes
scripts/
  init-rls.ts                 One-shot: apply policies.sql via DATABASE_URL
middleware.ts                 Wires Supabase middleware
```

## Schema notes

- Edges are polymorphic over `entity_kind` so we can connect any of
  {claim, experiment, run, todo} to any other.
- TipTap doc bodies live in `body_json` (JSONB) for fast diffing and
  realtime patching later.
- `agent_task` rows track currently-active agents (`stage`, `heartbeat_at`)
  for the Live view; rows transition to "completed" rather than getting
  deleted, so we keep history.
- `chat_session.agent_handle` stores the spawned Claude Code agent's
  process/session id so the sidecar can resume the right thread.

## Migrations

`pnpm db:generate` (after schema edits) → review SQL in `db/migrations/` →
commit. Apply with `pnpm db:push` (dev) or `pnpm db:migrate` (prod CI).
After every schema change that touches tables, re-run `pnpm db:init-rls`
so policies are applied to any new tables.
