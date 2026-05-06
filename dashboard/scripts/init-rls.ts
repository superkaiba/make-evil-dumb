import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import postgres from "postgres";

const url = process.env.DATABASE_URL;
if (!url) {
  console.error("DATABASE_URL is not set. Add it to dashboard/.env.local.");
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const sql = readFileSync(resolve(here, "..", "db", "policies.sql"), "utf8");
const client = postgres(url, { max: 1, prepare: false });

(async () => {
  try {
    await client.unsafe(sql);
    console.log("✅ RLS policies applied.");
  } catch (e) {
    console.error("❌ Failed to apply RLS policies:", e);
    process.exit(1);
  } finally {
    await client.end();
  }
})();
