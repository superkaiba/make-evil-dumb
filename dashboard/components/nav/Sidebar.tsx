"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Network, Activity, ListTodo, Calendar, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { href: "/graph", label: "Graph", icon: Network },
  { href: "/live", label: "Live", icon: Activity },
  { href: "/todos", label: "Todos", icon: ListTodo },
  { href: "/timeline/today", label: "Today", icon: Calendar },
  { href: "/timeline/week", label: "Week", icon: Calendar },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="panel flex h-full flex-col border-r p-3">
      <Link href="/graph" className="mb-4 block px-2 py-2 text-sm font-semibold tracking-tight">
        EPS Dashboard
      </Link>
      <nav className="flex flex-col gap-1">
        {items.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm",
                active
                  ? "bg-neutral-200 dark:bg-neutral-800"
                  : "hover:bg-neutral-100 dark:hover:bg-neutral-900",
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </Link>
          );
        })}
      </nav>
      <button
        type="button"
        className="mt-auto flex items-center gap-2 rounded-md border px-2 py-1.5 text-sm hover:bg-neutral-100 dark:hover:bg-neutral-900"
      >
        <Plus className="h-4 w-4" />
        New claim
      </button>
    </aside>
  );
}
