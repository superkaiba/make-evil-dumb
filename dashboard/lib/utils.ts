import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function confidenceClass(c: "HIGH" | "MODERATE" | "LOW" | null | undefined) {
  if (c === "HIGH") return "bg-confidence-high text-white";
  if (c === "MODERATE") return "bg-confidence-moderate text-black";
  if (c === "LOW") return "bg-confidence-low text-white";
  return "bg-neutral-300 text-black";
}
