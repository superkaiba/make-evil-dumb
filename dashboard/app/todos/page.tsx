import { desc } from "drizzle-orm";
import { getDb } from "@/db/client";
import { todos } from "@/db/schema";

export const dynamic = "force-dynamic";

export default async function TodosPage() {
  const db = getDb();
  const rows = await db
    .select({
      id: todos.id,
      text: todos.text,
      status: todos.status,
      githubIssueNumber: todos.githubIssueNumber,
      createdAt: todos.createdAt,
    })
    .from(todos)
    .orderBy(desc(todos.createdAt));

  return (
    <div className="mx-auto max-w-5xl p-6">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-xl font-semibold">Todos</h1>
        <span className="text-xs text-neutral-500">{rows.length} open</span>
      </header>

      {rows.length === 0 ? (
        <p className="text-sm text-neutral-500">No todos yet.</p>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {rows.map((t) => (
            <li
              key={t.id}
              className="panel flex items-start gap-3 rounded-md p-3 text-sm"
            >
              <input
                type="checkbox"
                disabled
                className="mt-1 h-3.5 w-3.5 cursor-not-allowed"
                aria-label="Done — wired up in M6"
              />
              <div className="flex-1">
                <div className="leading-snug">{t.text}</div>
                {t.githubIssueNumber != null && (
                  <a
                    href={`https://github.com/superkaiba/explore-persona-space/issues/${t.githubIssueNumber}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-1 inline-block text-[11px] text-neutral-500 hover:underline"
                  >
                    #{t.githubIssueNumber} ↗
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
