import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/nav/Sidebar";
import { ChatRail } from "@/components/chat/ChatRail";

export const metadata: Metadata = {
  title: "EPS Dashboard",
  description: "Research dashboard for explore-persona-space",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <div className="grid h-screen grid-cols-[240px_1fr_360px]">
          <Sidebar />
          <main className="overflow-y-auto">{children}</main>
          <ChatRail />
        </div>
      </body>
    </html>
  );
}
