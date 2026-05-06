/**
 * Backfill Supabase from GitHub issues:
 *   - clean-results-labeled issues  -> claim rows + figure (hero) + edges (derives_from for each Source issue)
 *   - status:proposed open issues   -> todo rows
 *
 * Idempotent: re-running upserts on github_issue_number unique key.
 */

import { execSync } from "node:child_process";
import { config } from "dotenv";
import postgres from "postgres";

config({ path: ".env.local" });

const url = process.env.DATABASE_URL;
if (!url) {
  console.error("DATABASE_URL not set in dashboard/.env.local");
  process.exit(1);
}

const sql = postgres(url, { prepare: false, max: 5 });

interface GhIssue {
  number: number;
  title: string;
  body: string;
  createdAt: string;
  updatedAt: string;
  state?: string;
}

function gh<T = GhIssue[]>(args: string[]): T {
  const out = execSync(["gh", ...args].join(" "), {
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
  return JSON.parse(out) as T;
}

function parseConfidence(title: string): "HIGH" | "MODERATE" | "LOW" | null {
  const m = title.match(/\((HIGH|MODERATE|LOW)\s+confidence\)\s*$/i);
  return m ? (m[1].toUpperCase() as "HIGH" | "MODERATE" | "LOW") : null;
}

function stripConfidenceSuffix(title: string): string {
  return title.replace(/\s*\((HIGH|MODERATE|LOW)\s+confidence\)\s*$/i, "").trim();
}

function parseHeroFigure(body: string): string | null {
  const lower = body.toLowerCase();
  const idx = lower.indexOf("### results");
  const start = idx >= 0 ? idx : 0;
  const m = body.slice(start).match(/!\[[^\]]*\]\((https?:\/\/[^\s)]+)\)/);
  return m ? m[1] : null;
}

function parseSourceIssueRefs(body: string): number[] {
  const head = body.match(/##\s+Source issues/i);
  if (!head) return [];
  const start = body.indexOf(head[0]) + head[0].length;
  const after = body.slice(start);
  const next = after.match(/\n##\s/);
  const section = next ? after.slice(0, next.index) : after;
  const nums = new Set<number>();
  for (const m of section.matchAll(/#(\d{1,6})\b/g)) {
    const n = parseInt(m[1], 10);
    if (!Number.isNaN(n)) nums.add(n);
  }
  return [...nums];
}

async function backfillClaims() {
  const issues = gh<GhIssue[]>([
    "issue", "list",
    "--label", "clean-results",
    "--state", "all",
    "--limit", "200",
    "--json", "number,title,body,createdAt,updatedAt,state",
  ]);
  console.log(`📥  ${issues.length} clean-results issues fetched`);

  const claimByIssue = new Map<number, string>();

  for (const iss of issues) {
    const confidence = parseConfidence(iss.title);
    const title = stripConfidenceSuffix(iss.title);
    const body = iss.body ?? "";
    const bodyJson = { kind: "markdown", text: body };

    const [row] = await sql<{ id: string }[]>`
      INSERT INTO claim (title, confidence, status, body_json, github_issue_number, created_at, updated_at)
      VALUES (
        ${title},
        ${confidence},
        ${"finalized"},
        ${sql.json(bodyJson)},
        ${iss.number},
        ${iss.createdAt},
        ${iss.updatedAt}
      )
      ON CONFLICT (github_issue_number) DO UPDATE SET
        title = EXCLUDED.title,
        confidence = EXCLUDED.confidence,
        body_json = EXCLUDED.body_json,
        updated_at = EXCLUDED.updated_at
      RETURNING id
    `;
    claimByIssue.set(iss.number, row.id);

    // Hero figure
    const heroUrl = parseHeroFigure(body);
    if (heroUrl) {
      const [fig] = await sql<{ id: string }[]>`
        INSERT INTO figure (url, caption, entity_kind, entity_id)
        VALUES (${heroUrl}, ${"hero (auto-extracted)"}, ${"claim"}, ${row.id})
        ON CONFLICT DO NOTHING
        RETURNING id
      `;
      if (fig) {
        await sql`UPDATE claim SET hero_figure_id = ${fig.id} WHERE id = ${row.id}`;
      }
    }
  }

  console.log(`✅ ${claimByIssue.size} claims upserted`);

  // Edges: derives_from for each Source issue ref
  let edgeCount = 0;
  for (const iss of issues) {
    const fromId = claimByIssue.get(iss.number);
    if (!fromId) continue;
    const refs = parseSourceIssueRefs(iss.body ?? "");
    for (const ref of refs) {
      const toId = claimByIssue.get(ref);
      if (!toId || toId === fromId) continue;
      await sql`
        INSERT INTO edge (from_kind, from_id, to_kind, to_id, type)
        VALUES (${"claim"}, ${fromId}, ${"claim"}, ${toId}, ${"derives_from"})
        ON CONFLICT DO NOTHING
      `;
      edgeCount++;
    }
  }
  console.log(`✅ ${edgeCount} derives_from edges`);
}

async function backfillTodos() {
  const issues = gh<GhIssue[]>([
    "issue", "list",
    "--label", "status:proposed",
    "--state", "open",
    "--limit", "200",
    "--json", "number,title,body,createdAt,updatedAt",
  ]);
  console.log(`📥  ${issues.length} status:proposed issues fetched`);

  let count = 0;
  for (const iss of issues) {
    const text = `#${iss.number} — ${iss.title}`;
    await sql`
      INSERT INTO todo (text, status, github_issue_number, created_at)
      VALUES (${text}, ${"open"}, ${iss.number}, ${iss.createdAt})
      ON CONFLICT (github_issue_number) DO UPDATE SET
        text = EXCLUDED.text
    `;
    count++;
  }
  console.log(`✅ ${count} todos upserted`);
}

(async () => {
  try {
    await backfillClaims();
    await backfillTodos();
    console.log("\n🎉 backfill complete");
  } catch (e) {
    console.error("backfill failed:", e);
    process.exit(1);
  } finally {
    await sql.end();
  }
})();
