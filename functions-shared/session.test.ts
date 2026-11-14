// Run: node --experimental-strip-types --test functions-shared/session.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { checkSession, makeSession, readCookie, sameKey, SESSION_DAYS } from "./session.ts";

test("sameKey: exact match only; unset key never matches", async () => {
  assert.equal(await sameKey("dev-key-123", "dev-key-123", SECRET), true);
  assert.equal(await sameKey("dev-key-124", "dev-key-123", SECRET), false);
  assert.equal(await sameKey("dev-key-123", undefined, SECRET), false);
  assert.equal(await sameKey("", "", SECRET), false);
});

const SECRET = "test-secret";

test("a fresh session verifies; wrong secret, tampering and expiry do not", async () => {
  const now = Date.UTC(2026, 10, 14);
  const v = await makeSession(SECRET, now);
  assert.equal(await checkSession(v, SECRET, now), true);
  assert.equal(await checkSession(v, "other", now), false);
  assert.equal(await checkSession(v.slice(0, -2) + "xx", SECRET, now), false);
  assert.equal(await checkSession(v, SECRET, now + (SESSION_DAYS + 1) * 864e5), false);
  assert.equal(await checkSession(undefined, SECRET, now), false);
  assert.equal(await checkSession(v, undefined, now), false);
});

test("readCookie finds the session among other cookies", () => {
  const r = new Request("https://x/", { headers: { Cookie: "a=1; ss_session=abc.def; b=2" } });
  assert.equal(readCookie(r), "abc.def");
});
