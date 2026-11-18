// 服务端 AI 调用 —— 复刻 APP 本地路径的非流式行为（含主→副 fallback、429 退避）。
// 永远 stream:false（中继不需要流式，结果整条进 outbox）。

import {
  getApiConfig,
  extractStreamReasoning,
  extractReasoning,
} from "./apiConfigs.js";
import {
  buildChatEndpoint,
  buildApiHeaders,
  buildChatRequestBody,
  assertSafeApiUrl,
} from "./requestBuilder.js";
import {
  parseServiceAccount,
  isVertexUrl,
  getAccessToken,
  buildVertexEndpoint,
  buildVertexBody,
  extractVertexDelta,
  extractVertexContent,
  extractVertexParts,
  describeVertexError,
} from "./vertex.js";

// 空闲超时（不是总时长）：连续这么久一个字节都没收到才判定连接挂死。
// 早期是 180s 总超时 → GLM 等长思考模型推理超过 3 分钟就被后端自己掐断，token 已扣却拿不到结果，
// 手机端再回退本地 = 同一轮扣两次费。流式下思考期间 reasoning delta 持续到达，不会误触。
const IDLE_TIMEOUT_MS = 600_000;
// 原生推理可能很长（数万字），outbox 存储有体积上限 → 封顶；手机端只用于思维链面板展示
const REASONING_CAP = 20_000;

function capReasoning(text) {
  if (typeof text !== "string") return null;
  const t = text.trim();
  if (!t) return null;
  return t.length > REASONING_CAP ? t.slice(0, REASONING_CAP) + "…" : t;
}

async function callOnce({ apiUrl, apiKey, model, apiType, messages, temperature, reasoningEffort, maxTokens, }) {
  assertSafeApiUrl(apiUrl);

  // 🅖 Vertex AI 直连分支：地址是 aiplatform.googleapis.com，或 key 是服务账号凭证 JSON。
  // APP 本地就是这么走的（凭证换 access_token → publishers/google/models 原生端点），
  // 中继以前没实现 → 会拼成 /v1/chat/completions 被 Google 404 且 body 为空。
  const sa = parseServiceAccount(apiKey);
  const isVertex = isVertexUrl(apiUrl) || !!sa;

  let endpoint;
  let headers;
  let body;
  if (isVertex) {
    const built = buildVertexEndpoint(apiUrl, sa, model);
    // 有服务账号 → 签 JWT 换 token；否则当作手机端已经填好 access_token
    const token = sa ? await getAccessToken(sa) : String(apiKey || "").trim();
    if (!token) throw new Error("Vertex: 缺少访问令牌（API key 为空）");
    endpoint = built.endpoint;
    headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    };
    body = built.openaiCompatible
      ? // Vertex 的 /endpoints/openapi 吃标准 OpenAI 体，只是鉴权换成 Bearer token
        buildChatRequestBody({
          apiUrl: "https://vertex-openai-compat.invalid",
          model,
          messages,
          temperature,
          reasoningEffort,
          stream: true,
          maxTokens,
        })
      : buildVertexBody({
          messages,
          temperature,
          maxTokens,
          model,
          reasoningEffort,
        });
  } else {
    endpoint = buildChatEndpoint(apiUrl);
    headers = buildApiHeaders(apiUrl, apiKey);
    // ⚠️ 用流式调 AI（stream:true）：部分 AI 代理（如 gemini 反代）对「非流式 + 图片」会 500，
    // 流式正常。后端在请求内读完整个 SSE 流、把 delta 拼成完整 content 再返回 —— 对手机端
    // 仍是「整条结果进 outbox」的非流式交付，只是后端内部走流式绕开代理的非流式限制。
    body = buildChatRequestBody({
      apiUrl,
      model,
      messages,
      temperature,
      reasoningEffort,
      stream: true,
      maxTokens,
    });
  }

  const controller = new AbortController();
  let timer = null;
  const kick = () => {
    clearTimeout(timer);
    timer = setTimeout(() => controller.abort(), IDLE_TIMEOUT_MS);
  };
  kick();
  let res;
  try {
    res = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      // Vertex 的报错结构是 {error:{code,message,status}}，直接拍平成一句人话
      const err = new Error(
        isVertex
          ? describeVertexError(res.status, errText)
          : `AI HTTP ${res.status}: ${errText.slice(0, 500)}`
      );
      err.status = res.status;
      throw err;
    }

    const config = getApiConfig(apiType);
    const ct = res.headers.get("content-type") || "";

    // 非 SSE（代理忽略了 stream，直接返回完整 JSON）→ 按非流式解析兜底
    if (!ct.includes("text/event-stream")) {
      const rawText = await res.text();
      let data;
      try {
        data = JSON.parse(rawText);
      } catch {
        data = rawText;
      }
      const content = isVertex
        ? extractVertexContent(data)
        : config.extractContent(data);
      if (content == null || content === "") {
        const err = new Error(
          "AI returned empty content (non-stream fallback)"
        );
        err.status = res.status;
        err.detail =
          typeof data === "string"
            ? data.slice(0, 300)
            : JSON.stringify(data).slice(0, 300);
        throw err;
      }
      const reasoning = isVertex
        ? extractVertexParts(Array.isArray(data) ? data[data.length - 1] : data)
            .reasoning
        : extractReasoning(data);
      return { content, reasoning: capReasoning(reasoning) };
    }

    // SSE 流式：逐行读 data:，累积 delta（正文 + 原生推理分开累积）
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let content = "";
    let reasoning = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      kick(); // 收到数据 → 重置空闲计时
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // 末行可能不完整，留到下次
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        let json;
        try {
          json = JSON.parse(payload);
        } catch {
          continue;
        }
        if (isVertex) {
          // Vertex 的 parts 可能一次给多段，且 thinking 模型会把思维链混在 parts 里
          const d = extractVertexDelta(json);
          if (d.text) content += d.text;
          if (d.reasoning) reasoning += d.reasoning;
        } else {
          const delta = config.extractStreamDelta(json);
          if (delta) content += delta;
          const rDelta = extractStreamReasoning(json);
          if (rDelta) reasoning += rDelta;
        }
      }
    }
    if (!content || !content.trim()) {
      const err = new Error("AI returned empty content (stream)");
      err.status = res.status;
      throw err;
    }
    return { content, reasoning: capReasoning(reasoning) };
  } finally {
    clearTimeout(timer);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** * 跑一次完整生成：主 API（带 429 退避重试）→ 失败且开了副 API fallback → 副 API。 * @returns {Promise<{content:string, reasoning:string|null}>} 原始模型文本（含 tag，手机端解析）+ 原生推理 */
export async function runGeneration(settings, messages, maxTokens) {
  const {
    mainApiUrl,
    mainApiKey,
    mainApiModel,
    apiType = "openai",
    temperature,
    reasoningEffort,
    autoRetryEnabled = true,
    maxRetries = 1,
    secondaryFallbackEnabled = true,
    secondaryApiUrl,
    secondaryApiKey,
    secondaryApiModel,
  } = settings || {};

  if (!mainApiUrl || !mainApiKey)
    throw new Error("settings.mainApiUrl / mainApiKey missing");

  const retries = autoRetryEnabled
    ? Math.max(0, Math.min(3, Number(maxRetries) || 0))
    : 0;

  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) await sleep(Math.min(1000 * 2 ** (attempt - 1), 30_000));
    try {
      return await callOnce({
        apiUrl: mainApiUrl,
        apiKey: mainApiKey,
        model: mainApiModel,
        apiType,
        messages,
        temperature,
        reasoningEffort,
        maxTokens,
      });
    } catch (e) {
      lastErr = e;
      // 只对 429 退避重试，其余错误立即跳出（避免重复扣费）
      if (e.status !== 429) break;
    }
  }

  // 主 API 失败 → 副 API fallback
  if (secondaryFallbackEnabled && secondaryApiUrl && secondaryApiKey) {
    try {
      return await callOnce({
        apiUrl: secondaryApiUrl,
        apiKey: secondaryApiKey,
        model: secondaryApiModel || mainApiModel,
        apiType,
        messages,
        temperature,
        reasoningEffort,
        maxTokens,
      });
    } catch (e) {
      lastErr = e;
    }
  }

  throw lastErr || new Error("AI generation failed");
}