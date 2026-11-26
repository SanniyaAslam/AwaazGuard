"use client";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Check a voice" },
  { href: "/dashboard", label: "Operations" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Main">
      {LINKS.map((l) => (
        <a key={l.href} href={l.href} aria-current={path === l.href ? "page" : undefined}>
          {l.label}
        </a>
      ))}
    </nav>
  );
}
