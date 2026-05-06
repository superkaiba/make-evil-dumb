"use client";

import { useState, type FormEvent } from "react";
import { createClient } from "@/lib/supabase/client";

export default function LoginPage() {
  const supabase = createClient();
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!email.trim()) return;
    setStatus("sending");
    setErrorMsg(null);
    const origin =
      typeof window !== "undefined" ? window.location.origin : process.env.NEXT_PUBLIC_SITE_URL;
    const { error } = await supabase.auth.signInWithOtp({
      email: email.trim(),
      options: { emailRedirectTo: `${origin}/api/auth/callback` },
    });
    if (error) {
      setStatus("error");
      setErrorMsg(error.message);
      return;
    }
    setStatus("sent");
  }

  return (
    <div className="flex h-full items-center justify-center">
      <div className="panel w-[360px] rounded-lg p-6">
        <h1 className="mb-1 text-lg font-semibold">EPS Dashboard</h1>
        <p className="mb-6 text-sm text-neutral-500">
          Sign in to edit. Anyone can browse without an account.
        </p>

        {status === "sent" ? (
          <div className="rounded-md border border-confidence-high/40 bg-confidence-high/10 p-3 text-sm">
            Check your inbox at <strong>{email}</strong> — click the link to sign in.
          </div>
        ) : (
          <form onSubmit={onSubmit} className="flex flex-col gap-3">
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-neutral-500">Email</span>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="rounded-md border bg-transparent px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-neutral-400"
              />
            </label>
            <button
              type="submit"
              disabled={status === "sending"}
              className="rounded-md bg-neutral-900 px-3 py-2 text-sm font-medium text-white hover:bg-neutral-800 disabled:opacity-50"
            >
              {status === "sending" ? "Sending…" : "Send magic link"}
            </button>
            {status === "error" && errorMsg && (
              <div className="text-xs text-red-600">{errorMsg}</div>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
