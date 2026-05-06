-- Row-level security policies.
--
-- Public dashboard: anyone (anon role) can SELECT entity rows.
-- Only authenticated users (you, signed in via GitHub OAuth) can INSERT/UPDATE/DELETE.
-- Chat sessions and chat messages are auth-only (read AND write) since chat is private.
--
-- Apply once after `pnpm db:push` via `pnpm db:init-rls`.
-- Idempotent: safe to re-run (DROP IF EXISTS, then CREATE).

-- ── enable RLS on every table ─────────────────────────────────────────────
ALTER TABLE claim          ENABLE ROW LEVEL SECURITY;
ALTER TABLE experiment     ENABLE ROW LEVEL SECURITY;
ALTER TABLE run            ENABLE ROW LEVEL SECURITY;
ALTER TABLE todo           ENABLE ROW LEVEL SECURITY;
ALTER TABLE edge           ENABLE ROW LEVEL SECURITY;
ALTER TABLE figure         ENABLE ROW LEVEL SECURITY;
ALTER TABLE comment        ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_task     ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_session   ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_message   ENABLE ROW LEVEL SECURITY;

-- ── public-readable entity tables ─────────────────────────────────────────
DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'claim', 'experiment', 'run', 'todo',
    'edge', 'figure', 'comment', 'agent_task'
  ] LOOP
    EXECUTE format('DROP POLICY IF EXISTS public_read ON %I', t);
    EXECUTE format(
      'CREATE POLICY public_read ON %I FOR SELECT TO anon, authenticated USING (true)', t
    );
    EXECUTE format('DROP POLICY IF EXISTS auth_write_insert ON %I', t);
    EXECUTE format(
      'CREATE POLICY auth_write_insert ON %I FOR INSERT TO authenticated WITH CHECK (true)', t
    );
    EXECUTE format('DROP POLICY IF EXISTS auth_write_update ON %I', t);
    EXECUTE format(
      'CREATE POLICY auth_write_update ON %I FOR UPDATE TO authenticated USING (true) WITH CHECK (true)', t
    );
    EXECUTE format('DROP POLICY IF EXISTS auth_write_delete ON %I', t);
    EXECUTE format(
      'CREATE POLICY auth_write_delete ON %I FOR DELETE TO authenticated USING (true)', t
    );
  END LOOP;
END $$;

-- ── chat tables: authenticated-only (read AND write) ──────────────────────
DROP POLICY IF EXISTS auth_only_chat_session ON chat_session;
CREATE POLICY auth_only_chat_session ON chat_session
  FOR ALL TO authenticated USING (true) WITH CHECK (true);

DROP POLICY IF EXISTS auth_only_chat_message ON chat_message;
CREATE POLICY auth_only_chat_message ON chat_message
  FOR ALL TO authenticated USING (true) WITH CHECK (true);
