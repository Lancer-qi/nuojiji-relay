// Vertex AI 直连支持。
//
// 背景：APP 的「API 设置」允许把地址填成 https://aiplatform.googleapis.com、
// key 填 Google Cloud 服务账号 JSON（凭证）。APP 本地会把凭证换成 access_token，
// 再走 publishers/google/models/{model}:generateContent 原生端点。
//
// 中继（本仓库）早期只支持 OpenAI 兼容地址，遇到 aiplatform 会拼成
// /v1/chat/completions → Google 返回 404（且 body 为空）→ 手机端只能回退本地生成。
// 本文件补齐这条链路：
// 1. parseServiceAccount：识别「key 是服务账号 JSON」
// 2. getAccessToken：RS256 签 JWT → oauth2.googleapis.com 换 token（带缓存）
// 3. buildVertexEndpoint：拼 Vertex 原生端点（含 region / project 推断）
// 4. buildVertexBody：OpenAI messages → Vertex contents/parts（含图片 inlineData）
// 5. extractVertexDelta / extractVertexContent：解析 Vertex 响应（含 thinking）
//
// 兼容两种填法：
// A. 只填 host（https://aiplatform.googleapis.com 或 https://us-central1-aiplatform.googleapis.com）
// → project 从凭证的 project_id 取，location 从 host 前缀取（无前缀 = global）
// B. 填完整资源路径（.../projects/P/locations/L/publishers/google/models/M）
// → 原样使用，只补 :streamGenerateContent?alt=sse
// C. 填 OpenAI 兼容端点（.../locations/L/endpoints/openapi）
// → 只补 /chat/completions，body 保持 OpenAI 格式

const DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token";
const CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
// token 有效期 1h；提前 60s 视为过期，避免请求边界上正好失效
const TOKEN_SAFETY_MS = 60_000;

// Workers 的 isolate 会复用，模块级缓存能跨请求命中；被回收也只是重新换一次 token。
const tokenCache = new Map();

/** key 是不是服务账号凭证 JSON（APP 里那个「凭证」输入框）。不是则返回 null。 */
export function parseServiceAccount(key) {
  if (typeof key !== "string") return null;
  const s = key.trim();
  if (!s.startsWith("{")) return null;
  try {
    const o = JSON.parse(s);
    if (!o || typeof o !== "object") return null;
    if (!o.client_email || !o.private_key) return null;
    return o;
  } catch {
    return null;
  }
}

/** 地址是不是 Vertex AI 的 host。 */
export function isVertexUrl(apiUrl) {
  return /(^|\/\/|\.)aiplatform\.googleapis\.com/i.test(String(apiUrl || ""));
}

/** 这次调用要不要走 Vertex 分支：地址是 aiplatform，或 key 是服务账号凭证。 */
export function isVertexRequest(apiUrl, apiKey) {
  return isVertexUrl(apiUrl) || !!parseServiceAccount(apiKey);
}

function pemToDer(pem) {
  const b64 = String(pem)
    .replace(/-----[^-]+-----/g, "")
    .replace(/\s+/g, "");
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function b64urlFromBytes(bytes) {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64urlFromString(str) {
  return b64urlFromBytes(new TextEncoder().encode(str));
}

async function signJwt(sa) {
  const now = Math.floor(Date.now() / 1000);
  const header = {
    alg: "RS256",
    typ: "JWT",
    ...(sa.private_key_id ? { kid: sa.private_key_id } : {}),
  };
  const payload = {
    iss: sa.client_email,
    scope: CLOUD_PLATFORM_SCOPE,
    aud: sa.token_uri || DEFAULT_TOKEN_URI,
    iat: now,
    exp: now + 3600,
  };
  const signingInput = `${b64urlFromString( JSON.stringify(header) )}.${b64urlFromString(JSON.stringify(payload))}`;

  const key = await crypto.subtle.importKey(
    "pkcs8",
    pemToDer(sa.private_key),
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign(
    "RSASSA-PKCS1-v1_5",
    key,
    new TextEncoder().encode(signingInput)
  );
  return `${signingInput}.${b64urlFromBytes(new Uint8Array(sig))}`;
}

/** * 服务账号 → OAuth access_token。同一凭证在有效期内复用缓存。 * @param {object} sa 服务账号 JSON * @returns {Promise<string>} access_token */
export async function getAccessToken(sa) {
  const cacheKey = `${sa.client_email}|${sa.private_key_id || ""}`;
  const hit = tokenCache.get(cacheKey);
  if (hit && hit.exp - TOKEN_SAFETY_MS > Date.now()) return hit.token;

  const jwt = await signJwt(sa);
  const res = await fetch(sa.token_uri || DEFAULT_TOKEN_URI, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body:
      "grant_type=" +
      encodeURIComponent("urn:ietf:params:oauth:grant-type:jwt-bearer") +
      "&assertion=" +
      encodeURIComponent(jwt),
  });
  const text = await res.text();
  if (!res.ok) {
    // 常见原因：私钥被截断 / 时钟偏差 / 服务账号被禁用 / 未开通 Vertex AI API
    throw new Error(`Vertex token ${res.status}: ${text.slice(0, 400)}`);
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Vertex token: 返回非 JSON: ${text.slice(0, 200)}`);
  }
  if (!data.access_token)
    throw new Error(
      `Vertex token: 响应里没有 access_token: ${text.slice(0, 200)}`
    );

  const ttl = Number(data.expires_in) || 3600;
  tokenCache.set(cacheKey, {
    token: data.access_token,
    exp: Date.now() + ttl * 1000,
  });
  return data.access_token;
}

/** 测试用：清空 token 缓存。 */
export function clearTokenCache() {
  tokenCache.clear();
}

function stripModelPrefix(model) {
  return String(model || "")
    .replace(/^publishers\/[^/]+\/models\//, "")
    .replace(/^google\//, "")
    .replace(/^models\//, "")
    .trim();
}

/** * 拼出真正要请求的端点。 * @returns {{endpoint:string, openaiCompatible:boolean}} */
export function buildVertexEndpoint(apiUrl, sa, model) {
  const base = String(apiUrl || "https://aiplatform.googleapis.com").replace(
    /\/+$/,
    ""
  );
  const modelId = stripModelPrefix(model);

  // C. 已经是 OpenAI 兼容端点 → 只补 /chat/completions
  if (/\/endpoints\/openapi$/.test(base)) {
    return { endpoint: `${base}/chat/completions`, openaiCompatible: true };
  }

  // B. 已经写全了模型资源路径
  if (/\/publishers\/[^/]+\/models\/[^/:]+/.test(base)) {
    const clean = base
      .replace(/:streamGenerateContent.*$/, "")
      .replace(/:generateContent.*$/, "");
    return {
      endpoint: `${clean}:streamGenerateContent?alt=sse`,
      openaiCompatible: false,
    };
  }

  // A. 只给了 host（可能带 region 前缀）
  const hostMatch = base.match(
    /^https?:\/\/(?:([a-z0-9-]+)-)?aiplatform\.googleapis\.com/i
  );
  const hostRegion = (hostMatch && hostMatch[1]) || "";
  const urlLocation = (base.match(/\/locations\/([^/]+)/) || [])[1] || "";
  const location = urlLocation || hostRegion || "global";

  const project =
    (base.match(/\/projects\/([^/]+)/) || [])[1] || (sa && sa.project_id) || "";
  if (!project) {
    throw new Error(
      "Vertex: 找不到 project_id（凭证 JSON 里没有 project_id，地址里也没写 /projects/xxx）"
    );
  }
  if (!modelId) throw new Error("Vertex: 缺少模型名");

  const host =
    location === "global"
      ? "https://aiplatform.googleapis.com"
      : `https://${location}-aiplatform.googleapis.com`;

  return {
    endpoint:
      `${host}/v1/projects/${project}/locations/${location}` +
      `/publishers/google/models/${modelId}:streamGenerateContent?alt=sse`,
    openaiCompatible: false,
  };
}

// ---------- 请求体：OpenAI messages → Vertex contents ----------

function contentToParts(content) {
  const parts = [];
  if (content == null) return parts;
  if (typeof content === "string") {
    if (content) parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) {
    if (typeof content === "object") {
      if (typeof content.text === "string" && content.text)
        parts.push({ text: content.text });
      else if (content.content != null) return contentToParts(content.content);
    }
    return parts;
  }
  for (const item of content) {
    if (!item) continue;
    if (typeof item === "string") {
      if (item) parts.push({ text: item });
      continue;
    }
    if (typeof item !== "object") continue;
    if (item.type === "text" || typeof item.text === "string") {
      if (item.text) parts.push({ text: item.text });
      continue;
    }
    if (item.type === "image_url" || item.image_url || item.type === "image") {
      const url = item.image_url?.url || item.url || item.source?.url || "";
      const m = String(url).match(/^data:([^;]+);base64,(.+)$/);
      if (m) {
        parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      } else if (url) {
        // 远端图片：Vertex 用 fileData 引用（需要能公网访问的 GCS/公开 URL）
        parts.push({
          fileData: {
            mimeType: item.image_url?.mime_type || "image/jpeg",
            fileUri: url,
          },
        });
      }
    }
  }
  return parts;
}

function reasoningBudget(effort, model) {
  const e = String(effort || "").toLowerCase();
  if (!e || e === "none" || e === "off") return null;
  const table = {
    minimal: 512,
    low: 1024,
    medium: 4096,
    high: 16384,
    max: 24576,
  };
  const v = table[e];
  if (!v) return null;
  // gemini-2.5-pro 不允许 thinkingBudget=0，但正数没问题
  return v;
}

/** * OpenAI 风格 messages → Vertex 原生请求体。 * @returns {object} */
export function buildVertexBody({ messages, temperature, maxTokens, model, reasoningEffort, }) {
  const list = Array.isArray(messages) ? messages : [];
  const systemTexts = [];
  const contents = [];

  for (const msg of list) {
    if (!msg) continue;
    const role = String(msg.role || "").toLowerCase();
    const parts = contentToParts(msg.content);
    if (!parts.length) continue;

    if (role === "system") {
      const t = parts
        .filter((p) => p.text)
        .map((p) => p.text)
        .join("\n\n");
      if (t) systemTexts.push(t);
      continue;
    }
    contents.push({
      role: role === "assistant" ? "model" : "user",
      parts,
    });
  }

  // Vertex 要求 contents 非空，且首条一般是 user
  if (!contents.length)
    contents.push({ role: "user", parts: [{ text: "请开始回复。" }] });

  const generationConfig = {};
  if (typeof temperature === "number" && Number.isFinite(temperature)) {
    generationConfig.temperature = temperature;
  }
  if (maxTokens) {
    generationConfig.maxOutputTokens = Math.min(Number(maxTokens) || 0, 65536);
  }
  const budget = reasoningBudget(reasoningEffort, model);
  if (budget) generationConfig.thinkingConfig = { thinkingBudget: budget };

  return {
    contents,
    ...(systemTexts.length
      ? { systemInstruction: { parts: [{ text: systemTexts.join("\n\n") }] } }
      : {}),
    ...(Object.keys(generationConfig).length ? { generationConfig } : {}),
  };
}

// ---------- 响应解析 ----------

/** * 从 Vertex 的 candidates[0].content.parts 里拆出正文与思考内容。 * thinking 模型会把思维链放在 {thought:true, text} 的 part 里。 */
export function extractVertexParts(data) {
  const parts = data?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) return { text: "", reasoning: "" };
  let text = "";
  let reasoning = "";
  for (const p of parts) {
    if (!p || typeof p.text !== "string") continue;
    if (p.thought === true) reasoning += p.text;
    else text += p.text;
  }
  return { text, reasoning };
}

/** 非流式兜底：一次性 JSON。 */
export function extractVertexContent(data) {
  const d = Array.isArray(data) ? data[data.length - 1] : data;
  const { text } = extractVertexParts(d);
  if (text) return text;
  // OpenAI 兼容端点（/endpoints/openapi）走标准格式
  return d?.choices?.[0]?.message?.content ?? null;
}

/** 流式：单条 SSE data 里的增量。 */
export function extractVertexDelta(json) {
  const { text, reasoning } = extractVertexParts(json);
  if (text || reasoning) return { text, reasoning };
  const delta = json?.choices?.[0]?.delta;
  if (delta) {
    return {
      text: typeof delta.content === "string" ? delta.content : "",
      reasoning:
        typeof delta.reasoning_content === "string"
          ? delta.reasoning_content
          : "",
    };
  }
  return { text: "", reasoning: "" };
}

/** 上游报错时把 Google 的错误结构拍平成一句人话，便于手机端直接看。 */
export function describeVertexError(status, bodyText) {
  try {
    const o = JSON.parse(bodyText);
    const msg = o?.error?.message || o?.message;
    if (msg) return `Vertex ${status}: ${msg}`;
  } catch {
    /* 不是 JSON */
  }
  return `Vertex ${status}: ${ String(bodyText || "").slice(0, 400) || "(空响应)" }`;
}