/**
 * Paywall gate for /data/*: premium files and the history CSVs (they name every past and open
 * position) need a valid session cookie; everything else under /data passes through untouched.
 */
import { checkSession, readCookie } from "../../functions-shared/session";

type Env = { SESSION_SECRET?: string };
type Ctx = { request: Request; env: Env; next: () => Promise<Response> };

const GATED = ["/data/premium/", "/data/history/"];

export async function onRequest({ request, env, next }: Ctx): Promise<Response> {
  const path = new URL(request.url).pathname;
  if (!GATED.some((p) => path.startsWith(p))) return next();

  if (!(await checkSession(readCookie(request), env.SESSION_SECRET))) {
    return new Response(JSON.stringify({ error: "locked" }), {
      status: 401,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  }
  const res = await next();
  const out = new Response(res.body, res);
  out.headers.set("Cache-Control", "private, no-store"); // never cache paid data in shared caches
  return out;
}
