"use client";

import { useState } from "react";
import { ChevronRight, ChevronLeft, Send, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

export function ChatRail() {
  const [open, setOpen] = useState(true);
  const [draft, setDraft] = useState("");

  return (
    <aside
      className={cn(
        "panel flex h-full flex-col border-l transition-[width] duration-150",
        open ? "w-[360px]" : "w-12",
      )}
      style={{ width: open ? undefined : 48 }}
    >
      <div className="flex items-center justify-between border-b px-3 py-2">
        {open && (
          <div className="flex items-center gap-2 text-sm font-medium">
            <Sparkles className="h-4 w-4" />
            <span>Claude agent</span>
          </div>
        )}
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="rounded p-1 hover:bg-neutral-100 dark:hover:bg-neutral-900"
          aria-label={open ? "Collapse chat" : "Expand chat"}
        >
          {open ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </button>
      </div>

      {open && (
        <>
          <div className="flex-1 overflow-y-auto p-3 text-sm text-neutral-500">
            <p className="italic">
              Chat backend lands in milestone 7 (Cloudflare Tunnel + Python sidecar +
              Claude Agent SDK). This rail is the UI shell.
            </p>
          </div>

          <form
            className="flex items-end gap-2 border-t p-3"
            onSubmit={(e) => {
              e.preventDefault();
              setDraft("");
            }}
          >
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Ask about a claim, experiment, or the whole project…"
              rows={2}
              className="flex-1 resize-none rounded-md border bg-transparent px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-neutral-400"
            />
            <button
              type="submit"
              disabled={!draft.trim()}
              className="rounded-md border p-2 disabled:opacity-40 hover:bg-neutral-100 dark:hover:bg-neutral-900"
              aria-label="Send"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>
        </>
      )}
    </aside>
  );
}
