import { TOKEN_KEY, withToken } from "./lib.js";

export function getToken() {
  try {
    return window.localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function setToken(value) {
  try {
    if (value) window.localStorage.setItem(TOKEN_KEY, value);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Private mode: the token lives for this page only.
  }
}

export class AuthError extends Error {}

export async function api(path, options = {}) {
  const token = getToken();
  const headers = { ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    ...options,
    headers,
    body: options.json !== undefined ? JSON.stringify(options.json) : options.body,
  });
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text.slice(0, 240) };
    }
  }
  if (response.status === 401) {
    window.dispatchEvent(new CustomEvent("desk-auth-required"));
    throw new AuthError("This desk needs a token.");
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  return body;
}

export function authed(url) {
  return withToken(url, getToken());
}
