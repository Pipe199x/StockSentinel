/**
 * Paywall session cookie for the Cloudflare Pages Functions in ../functions.
 *
 * Value: base64url(JSON {exp}) + "." + base64url(HMAC-SHA256(payload, SESSION_SECRET)).
 * Stateless: no store needed; it expires after SESSION_DAYS, then the license key is
 * checked against Gumroad again. Kept outside functions/ so Pages does not route it.
 */
export const COOKIE = "ss_session";
export const SESSION_DAYS = 7;

const enc = new TextEncoder();

function b64url(bytes: ArrayBuffer | Uint8Array): string {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let s = "";
  for (const x of b) s += String.fromCharCode(x);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function fromB64url(s: string): Uint8Array {
  const b = atob(s.replace(/-/g, "+").replace(/_/g, "/") + "===".slice((s.length + 3) % 4));
  return Uint8Array.from(b, (c) => c.charCodeAt(0));
}

async function hmacKey(secret: string) {
  return crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

export async function makeSession(secret: string, now = Date.now()): Promise<string> {
  const payload = b64url(enc.encode(JSON.stringify({ exp: now + SESSION_DAYS * 864e5 })));
  const sig = await crypto.subtle.sign("HMAC", await hmacKey(secret), enc.encode(payload));
  return `${payload}.${b64url(sig)}`;
}

/** True only for an untampered, unexpired cookie value. */
export async function checkSession(value: string | null | undefined, secret: string | undefined, now = Date.now()): Promise<boolean> {
  if (!value || !secret) return false;
  const [payload, sig] = value.split(".");
  if (!payload || !sig) return false;
  try {
    const ok = await crypto.subtle.verify("HMAC", await hmacKey(secret), fromB64url(sig), enc.encode(payload));
    if (!ok) return false;
    const { exp } = JSON.parse(new TextDecoder().decode(fromB64url(payload)));
    return typeof exp === "number" && exp > now;
  } catch {
    return false;
  }
}

/**
 * Constant-time equality for a submitted key vs a configured one: compares HMACs of both, so
 * timing does not depend on where the strings differ (or on their lengths).
 */
export async function sameKey(given: string, expected: string | undefined, secret: string): Promise<boolean> {
  if (!expected) return false;
  const k = await hmacKey(secret);
  const [a, b] = await Promise.all([crypto.subtle.sign("HMAC", k, enc.encode(given)), crypto.subtle.sign("HMAC", k, enc.encode(expected))]);
  const x = new Uint8Array(a);
  const y = new Uint8Array(b);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

export function readCookie(request: Request, name = COOKIE): string | null {
  const raw = request.headers.get("Cookie") || "";
  for (const part of raw.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k === name) return v.join("=");
  }
  return null;
}

export function sessionCookie(value: string): string {
  return `${COOKIE}=${value}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=${SESSION_DAYS * 86400}`;
}

export const clearCookie = `${COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`;
