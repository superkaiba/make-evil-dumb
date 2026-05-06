import Link from "next/link";
import { desc } from "drizzle-orm";
import { getDb } from "@/db/client";
import { claims } from "@/db/schema";
import { confidenceClass } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function GraphPage() {
  const db = getDb();
  const rows = await db
    .select({
      id: claims.id,
      title: claims.title,
      confidence: claims.confidence,
      githubIssueNumber: claims.githubIssueNumber,
      updatedAt: claims.updatedAt,
    })
    .from(claims)
    .orderBy(desc(claims.updatedAt));

  return (
    <div className="mx-auto max-w-5xl p-6">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold">Claims</h1>
        <span className="text-xs text-neutral-500">
          {rows.length} total · graph view in M3
        </span>
      </header>

      {rows.length === 0 ? (
        <p className="text-sm text-neutral-500">No claims yet — run `pnpm db:backfill` on the VM.</p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((c) => (
            <li key={c.id}>
              <Link
                href={`/claim/${c.id}`}
                className="panel flex items-start gap-3 rounded-md p-3 transition-colors hover:bg-neutral-100 dark:hover:bg-neutral-900"
              >
                <span
                  className={`mt-0.5 inline-flex shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${confidenceClass(
                    c.confidence,
                  )}`}
                >
                  {c.confidence ?? "—"}
                </span>
                <div className="flex-1">
                  <div className="text-sm leading-snug">{c.title}</div>
                  <div className="mt-1 text-[11px] text-neutral-500">
                    {c.githubIssueNumber != null && <>#{c.githubIssueNumber} · </>}
                    {new Date(c.updatedAt).toLocaleDateString()}
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
