/**
 * POST /api/unlock {license_key} → checks the key with Gumroad and sets the session cookie.
 *
 * Accepted only if Gumroad confirms the purchase and it is not refunded, charged back, disputed,
 * or (for memberships) cancelled, ended or failing. increment_uses_count=false: checking a key
 * must not consume it.
 *
 * Env: SESSION_SECRET (required); GUMROAD_PRODUCT_ID, or GUMROAD_PRODUCT_PERMALINK for products
 * created before 2023 (Gumroad accepts one or the other); DEV_UNLOCK_KEY (optional owner key).
 */
import { makeSession, sameKey, sessionCookie } from "../../functions-shared/session";

type Env = { SESSION_SECRET?: string; GUMROAD_PRODUCT_ID?: string; GUMROAD_PRODUCT_PERMALINK?: string; DEV_UNLOCK_KEY?: string };
type Ctx = { request: Request; env: Env };

type Purchase = {
  refunded?: boolean;
  chargebacked?: boolean;
  disputed?: boolean;
  subscription_cancelled_at?: string | null;
  subscription_ended_at?: string | null;
  subscription_failed_at?: string | null;
};

const json = (status: number, body: unknown, extra: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store", ...extra } });

export async function onRequestPost({ request, env }: Ctx): Promise<Response> {
  if (!env.SESSION_SECRET || !(env.GUMROAD_PRODUCT_ID || env.GUMROAD_PRODUCT_PERMALINK || env.DEV_UNLOCK_KEY)) {
    return json(503, { error: "Unlocking is not configured yet." });
  }
  let key = "";
  try {
    key = String(((await request.json()) as { license_key?: unknown }).license_key ?? "").trim();
  } catch {
    return json(400, { error: "Send a license key." });
  }
  if (!key || key.length > 200) return json(400, { error: "Send a license key." });

  // Owner's developer key (env DEV_UNLOCK_KEY): same session, no Gumroad call. Delete the env var to revoke.
  if (await sameKey(key, env.DEV_UNLOCK_KEY, env.SESSION_SECRET)) {
    return json(200, { ok: true }, { "Set-Cookie": sessionCookie(await makeSession(env.SESSION_SECRET)) });
  }

  if (!(env.GUMROAD_PRODUCT_ID || env.GUMROAD_PRODUCT_PERMALINK)) return json(401, { error: "That license key is not active for StockSentinel." });
  const form = new URLSearchParams({ license_key: key, increment_uses_count: "false" });
  if (env.GUMROAD_PRODUCT_ID) form.set("product_id", env.GUMROAD_PRODUCT_ID);
  else form.set("product_permalink", env.GUMROAD_PRODUCT_PERMALINK!);

  let data: { success?: boolean; purchase?: Purchase } = {};
  try {
    const r = await fetch("https://api.gumroad.com/v2/licenses/verify", { method: "POST", body: form });
    data = await r.json();
  } catch {
    return json(502, { error: "Could not reach Gumroad. Try again in a minute." });
  }
  const p = data.purchase ?? {};
  const valid =
    data.success === true && !p.refunded && !p.chargebacked && !p.disputed && !p.subscription_cancelled_at && !p.subscription_ended_at && !p.subscription_failed_at;
  if (!valid) return json(401, { error: "That license key is not active for StockSentinel." });

  return json(200, { ok: true }, { "Set-Cookie": sessionCookie(await makeSession(env.SESSION_SECRET)) });
}
