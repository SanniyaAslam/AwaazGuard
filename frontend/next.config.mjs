/** @type {import('next').NextConfig} */
// Production (`npm run build`): static export to `out/`, served by the FastAPI gateway on the
// same origin, so the browser calls /api/* directly and no Node server is needed.
// Development (`npm run dev`): /api/* is proxied to the gateway at GATEWAY_URL.
const isDev = process.env.NODE_ENV === "development";
const gateway = process.env.GATEWAY_URL || "http://localhost:8000";

const nextConfig = isDev
  ? { async rewrites() { return [{ source: "/api/:path*", destination: `${gateway}/api/:path*` }]; } }
  : { output: "export", trailingSlash: true, images: { unoptimized: true } };

export default nextConfig;
