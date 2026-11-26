import "./globals.css";
import Nav from "@/components/Nav";

export const metadata = {
  title: "AwaazGuard",
  description: "Check whether a voice note or call recording is a real person or an AI voice clone.",
};

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
      <rect width="28" height="28" rx="7" fill="currentColor" />
      {[5, 9, 13, 17, 21].map((x, i) => (
        <rect key={x} x={x} y={14 - [4, 8, 10, 6, 3][i]} width="2.4"
              height={[4, 8, 10, 6, 3][i] * 2} rx="1.2" fill="var(--paper)" />
      ))}
    </svg>
  );
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&display=swap"
        />
      </head>
      <body>
        <header className="topbar">
          <div className="shell">
            <a className="brand" href="/"><Logo />AwaazGuard</a>
            <Nav />
          </div>
        </header>
        <main><div className="shell">{children}</div></main>
        <footer className="foot">
          <div className="shell">
            AwaazGuard gives a second opinion, not proof. If a call asks for money or codes, hang up and
            call the person back on a number you already know.
          </div>
        </footer>
      </body>
    </html>
  );
}
