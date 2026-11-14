/** POST /api/logout → clears the paywall session cookie. */
import { clearCookie } from "../../functions-shared/session";

export async function onRequestPost(): Promise<Response> {
  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store", "Set-Cookie": clearCookie },
  });
}
