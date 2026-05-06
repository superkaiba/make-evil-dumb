import Link from "next/link";
import Image from "next/image";
import { notFound } from "next/navigation";
import { eq } from "drizzle-orm";
import { getDb } from "@/db/client";
import { claims, figures } from "@/db/schema";
import { confidenceClass } from "@/lib/utils";

export const dynamic = "force-dynamic";

type BodyJson = { kind: "markdown"; text: string } | null;

export default async function ClaimPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const db = getDb();
  const [claim] = await db.select().from(claims).where(eq(claims.id, id)).limit(1);
  if (!claim) notFound();

  let hero: { url: string; caption: string | null } | null = null;
  if (claim.heroFigureId) {
    const [fig] = await db
      .select({ url: figures.url, caption: figures.caption })
      .from(figures)
      .where(eq(figures.id, claim.heroFigureId))
      .limit(1);
    if (fig) hero = fig;
  }

  const body = claim.bodyJson as BodyJson;
  const markdown = body?.text ?? "";

  return (
    <article className="mx-auto max-w-3xl px-6 py-8">
      <Link
        href="/graph"
        className="text-xs text-neutral-500 hover:underline"
      >
        ← all claims
      </Link>

      <div className="mt-3 flex items-start gap-3">
        <span
          className={`mt-1.5 inline-flex shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold ${confidenceClass(
            claim.confidence,
          )}`}
        >
          {claim.confidence ?? "—"}
        </span>
        <h1 className="text-2xl font-semibold leading-tight">{claim.title}</h1>
      </div>

      {claim.githubIssueNumber != null && (
        <div className="mt-2 text-xs text-neutral-500">
          <a
            href={`https://github.com/superkaiba/explore-persona-space/issues/${claim.githubIssueNumber}`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline"
          >
            #{claim.githubIssueNumber} on GitHub ↗
          </a>
        </div>
      )}

      {hero && (
        <figure className="mt-6">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={hero.url}
            alt={hero.caption ?? "Hero figure"}
            className="w-full rounded-lg border"
          />
          {hero.caption && (
            <figcaption className="mt-2 text-xs text-neutral-500">
              {hero.caption}
            </figcaption>
          )}
        </figure>
      )}

      <pre className="mt-6 whitespace-pre-wrap font-sans text-sm leading-relaxed">
        {markdown || "(empty body)"}
      </pre>
    </article>
  );
}
