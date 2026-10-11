import { reactive } from "vue";

export const session = reactive({ status: null, user: null });

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function messageFrom(data, status) {
  if (!data) return `Request failed (${status})`;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail.map((d) => `${(d.loc || []).slice(-1)[0] || "value"}: ${d.msg}`).join("; ");
  }
  return `Request failed (${status})`;
}

export async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, credentials: "same-origin", headers: {} };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    if (res.status === 401 && !path.startsWith("/api/auth/")) {
      session.user = null;
      window.location.assign("/login?next=" + encodeURIComponent(location.pathname));
    }
    throw new ApiError(res.status, messageFrom(data, res.status));
  }
  return data;
}

export async function loadStatus() {
  session.status = await api("/api/auth/status");
  session.user = session.status.user;
  return session.status;
}

export const BOARDS = [
  { value: "main", label: "Main" },
  { value: "side", label: "Sideboard" },
  { value: "commander", label: "Commander" },
  { value: "maybe", label: "Maybe" },
];

export const FORMATS = ["", "standard", "pioneer", "modern", "legacy", "vintage", "pauper", "commander",
  "brawl", "limited", "casual"];

export const FLAG_TEXT = {
  unmatched: "No matching card",
  low_confidence: "Low confidence",
};

const LANGUAGES = { de: "German", fr: "French", it: "Italian", es: "Spanish", pt: "Portuguese", ja: "Japanese",
  ko: "Korean", ru: "Russian", zhs: "Chinese", zht: "Chinese", he: "Hebrew", la: "Latin", grc: "Greek",
  ar: "Arabic", sa: "Sanskrit", ph: "Phyrexian" };

// How a card was identified, when it wasn't a plain English name match.
export function methodText(method) {
  if (!method || method === "name") return "";
  if (method === "printing") return "Found by set code and collector number";
  if (method === "alias") return "Found by alternate name";
  if (method === "choice") return "Picked from similar cards";
  if (method.startsWith("foreign:")) return `Found by ${LANGUAGES[method.slice(8)] || "foreign"} name`;
  return "";
}

// Flags about the number of copies are shown on the quantity box, not as a pill.
export const QTY_FLAGS = ["count_mismatch", "check_basic_count"];

export function pct(x) {
  return Math.round((x || 0) * 100) + "%";
}

export function fmtDate(iso) {
  return iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";
}
