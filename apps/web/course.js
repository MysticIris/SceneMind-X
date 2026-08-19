const $ = (id) => document.getElementById(id);
const DEBUG_UI = new URLSearchParams(window.location.search).get("debug") === "1";
document.documentElement.classList.toggle("debug-ui", DEBUG_UI);
const state = {
  assets: [],
  libraryAssets: [],
  localAssets: [],
  sessionAssets: [],
  libraries: [],
  libraryPage: 1,
  libraryPageCount: 1,
  libraryTotal: 0,
  libraryLoadSequence: 0,
  selectedRef: null,
  contextRefs: [],
  currentView: "chat",
  chatSelectionVersion: 0,
  chatQueue: Promise.resolve(),
  chatClientSequence: 0,
  chatPendingRefs: new Set(),
  conversations: [],
  currentConversationId: null,
  chatState: null,
  selectedSearchLabels: new Set(),
  contentTypeUserSelected: false,
  contentLengthUserEdited: false,
  contentLengthProfiles: {},
  canonicalPreview: [],
  provider: null,
  providerProfiles: null,
  providerIndexCoverage: null,
  pendingBackfillScope: null,
  activeBackfillTaskId: null,
  backfillPollTimer: null,
  backfillScanInProgress: false,
  backfillOperationId: null,
  backfillReturnFocus: null,
  toastTimer: null,
  onlineRecoveryRequested: false,
  providerRevalidateInFlight: null,
  providerSwitchSequence: 0,
  providerProjectionSequence: 0,
  retrievalProjectionSequence: 0,
  retrievalRuntime: null,
  batchSelections: {
    chat: new Set(),
    generation: new Set(),
    retrieval: new Set(),
    compare: new Set(),
  },
  workspaces: {
    generation: {workspace_id: "generation-default", version: 0, refs: [], queue: Promise.resolve(), clientSequence: 0, pendingRefs: new Set()},
    retrieval: {workspace_id: "retrieval-default", version: 0, refs: [], queue: Promise.resolve(), clientSequence: 0, pendingRefs: new Set()},
    compare: {workspace_id: "compare-default", version: 0, refs: [], queue: Promise.resolve(), clientSequence: 0, pendingRefs: new Set()},
  },
};

const LEGACY_FIELD_DISPLAY_NAMES = {
  global_observation: "整体观察",
  global_scene: "整体场景",
  subjects: "主要主体",
  main_subjects: "主要主体",
  activities: "动作",
  relations: "空间关系",
  attributes: "显著属性",
  visible_text_candidates: "文字候选（尚未验证）",
  visible_text: "可见文字",
  scene_hypotheses: "场景假设",
  uncertainties: "不确定信息",
  uncertainty: "不确定信息",
  evidence_descriptions: "证据描述",
  ocr_status: "OCR状态",
  verified_text_status: "文字验证状态",
};

const PUBLIC_STATUS_DISPLAY_NAMES = {
  pass: "检查通过",
  ready: "已就绪",
  warning: "存在提醒",
  needs_review: "需要复核",
  failed: "生成失败",
  pending: "待审核",
  machine_provisional: "机器暂定",
  image_only_unverified_text: "仅有图片证据，文字尚未验证",
  not_available: "未提供",
};

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const resultMediaUrl = (item) => (
  item?.thumbnail_url || item?.content_url || item?.image_url || ""
);

function resultMediaMarkup(item, alt, imageClass = "") {
  const mediaUrl = resultMediaUrl(item);
  const assetId = item?.asset_id || item?.source_asset_id || "unknown_asset";
  return `
    <div class="result-media-shell" data-media-shell="${escapeHtml(assetId)}">
      ${mediaUrl ? `<img class="${escapeHtml(imageClass)}" data-result-media data-media-url="${escapeHtml(mediaUrl)}" src="${escapeHtml(mediaUrl)}" alt="${escapeHtml(alt)}">` : ""}
      <div class="result-media-fallback ${mediaUrl ? "hidden" : ""}">
        <span>图片暂时无法显示</span>
        <small>${escapeHtml(assetId)}</small>
        <button type="button" data-media-retry>重新加载</button>
      </div>
    </div>`;
}

document.addEventListener("error", (event) => {
  const image = event.target;
  if (!(image instanceof HTMLImageElement) || !image.matches("[data-result-media]")) return;
  image.classList.add("hidden");
  image.closest("[data-media-shell]")?.querySelector(".result-media-fallback")?.classList.remove("hidden");
}, true);

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-media-retry]");
  if (!button) return;
  const shell = button.closest("[data-media-shell]");
  const image = shell?.querySelector("[data-result-media]");
  if (!image) return;
  const source = image.dataset.mediaUrl || "";
  if (!source) return;
  shell.querySelector(".result-media-fallback")?.classList.add("hidden");
  image.classList.remove("hidden");
  image.src = `${source}${source.includes("?") ? "&" : "?"}_media_retry=${Date.now()}`;
});

const pretty = (value) => {
  if (value === null || value === undefined) return "未提供";
  if (Array.isArray(value)) return value.length ? value.join("、") : "暂无";
  if (value === "") return "暂无";
  if (typeof value === "string") return PUBLIC_STATUS_DISPLAY_NAMES[value] || value;
  return typeof value === "object"
    ? Object.values(value).map(pretty).filter(Boolean).join("；") || "暂无"
    : String(value);
};

const PUBLIC_TEXT_FAILURE = "本次生成未成功，请稍后重试。";

function plainPublicText(value) {
  if (typeof value !== "string") return "";
  const text = value.trim();
  if (!text) return "";
  if (text.startsWith("{") || text.startsWith("[")) {
    try {
      const decoded = JSON.parse(text);
      if (decoded && typeof decoded === "object") return "";
    } catch (_) {
      // A normal sentence may begin with a bracket. Only valid JSON is blocked.
    }
  }
  return text;
}

function publicDisplayText(content, fallback = PUBLIC_TEXT_FAILURE) {
  const direct = plainPublicText(content);
  if (direct) return direct;
  if (!content || typeof content !== "object" || Array.isArray(content)) return fallback;

  for (const key of ["display_text", "final_text", "public_answer"]) {
    const value = plainPublicText(content[key]);
    if (value) return value;
  }

  for (const containerKey of ["answer", "content", "public_result"]) {
    const container = content[containerKey];
    const containerText = plainPublicText(container);
    if (containerText) return containerText;
    if (!container || typeof container !== "object" || Array.isArray(container)) continue;
    for (const key of ["display_text", "final_text", "public_answer", "answer", "content"]) {
      const value = plainPublicText(container[key]);
      if (value) return value;
    }
  }

  const resultData = content.result?.data;
  const finalOutput = plainPublicText(resultData?.final_output);
  if (finalOutput) return finalOutput;
  const parsed = resultData?.parsed_output;
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    for (const key of ["display_text", "final_text", "public_answer", "answer"]) {
      const value = plainPublicText(parsed[key]);
      if (value) return value;
    }
  }

  if (Array.isArray(content.results)) {
    return `已返回 ${content.results.length} 个检索结果（${plainPublicText(content.mode) || "retrieval"}）。`;
  }
  if (Array.isArray(content.ranking)) {
    return `已完成 ${content.ranking.length} 张图片排序，最佳结果已标出。`;
  }
  return fallback;
}

function showError(message = "") {
  $("error").textContent = message;
  $("error").classList.toggle("hidden", !message);
}

function clearToast() {
  if (state.toastTimer) window.clearTimeout(state.toastTimer);
  state.toastTimer = null;
  $("toast").textContent = "";
  $("toast").classList.add("hidden");
}

function showToast(message, duration = 2600) {
  clearToast();
  $("toast").textContent = message;
  $("toast").classList.remove("hidden");
  state.toastTimer = window.setTimeout(clearToast, duration);
}

function currentContentLengthProfile() {
  const contentType = $("content-mode")?.value || "auto";
  return state.contentLengthProfiles[contentType]
    || state.contentLengthProfiles.auto
    || null;
}

function applyContentLengthProfile({resetValue = true} = {}) {
  const input = $("content-length");
  const profile = currentContentLengthProfile();
  if (!input || !profile) return;
  input.min = String(profile.input_min);
  input.max = String(profile.input_max);
  if (resetValue) input.value = String(profile.default);
  const help = $("content-length-help");
  if (help) {
    help.textContent = `可设置 ${profile.input_min}–${profile.input_max} 字`;
  }
}

function normalizeContentLengthInput({notify = false} = {}) {
  const input = $("content-length");
  const profile = currentContentLengthProfile();
  if (!input || !profile) return null;
  const raw = String(input.value || "").trim();
  const parsed = raw ? Number(raw) : Number.NaN;
  let target = Number.isFinite(parsed) ? Math.round(parsed) : profile.default;
  const original = target;
  target = Math.min(profile.input_max, Math.max(profile.input_min, target));
  input.value = String(target);
  if (notify && (!Number.isFinite(parsed) || target !== original || parsed !== target)) {
    const contentType = $("content-mode")?.value || "auto";
    const message = original > profile.input_max
      && ["auto", "article"].includes(contentType)
      ? `该内容类型最多支持${profile.input_max}字，目标长度已调整为${target}字。`
      : `已将目标长度调整为 ${target} 字；当前类型可设置 ${profile.input_min}–${profile.input_max} 字。`;
    showToast(message);
  }
  return target;
}

async function loadContentLengthProfiles() {
  const payload = await api("/course/content-length-profiles");
  state.contentLengthProfiles = payload.profiles || {};
  applyContentLengthProfile({resetValue: true});
}

function publicApiErrorMessage(body, status) {
  const detail = typeof body?.detail === "string" ? body.detail.trim() : "";
  if (DEBUG_UI && detail) return detail;
  if (detail && /[\u3400-\u9fff]/u.test(detail)) return detail;
  if (detail === "public_top_k_must_be_between_1_and_5") {
    return "检索参数异常，请重新尝试。";
  }
  if (status === 400 || status === 422) {
    return "请求参数有误，请检查后重试。";
  }
  if (status === 409) return "当前状态已更新，请重新尝试。";
  if (status === 429) return "请求较多，请稍后重试。";
  if (status >= 500) return "服务暂时不可用，请稍后重试。";
  if (detail && !/^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/i.test(detail)) {
    return detail;
  }
  return `请求失败（HTTP ${status}），请稍后重试。`;
}

async function api(path, options = {}) {
  const {
    timeoutMs = 60000,
    signal: externalSignal,
    ...fetchOptions
  } = options;
  const controller = new AbortController();
  const relayAbort = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort();
    else externalSignal.addEventListener("abort", relayAbort, {once: true});
  }
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      headers: {"Content-Type": "application/json", ...(fetchOptions.headers || {})},
      ...fetchOptions,
      signal: controller.signal,
    });
    const body = await response.json().catch(() => ({detail: response.statusText}));
    if (!response.ok) {
      const error = new Error(publicApiErrorMessage(body, response.status));
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error("模型请求等待超时，按钮已恢复；图片、对话、工作区和已有结果均已保留。");
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
    externalSignal?.removeEventListener?.("abort", relayAbort);
  }
}

const PROVIDER_DISABLED_MESSAGE = "当前尚未接入模型。请先在顶部“模型接入”中选择本地、云端或服务器演示模式。";
const PROVIDER_PREFLIGHT_MESSAGE = "正在检查模型连接；如状态已过期，系统会自动执行一次有界恢复。";

function isExactRuntimeModelIdentityQuestion(value) {
  let normalized = String(value || "").trim();
  if (["？", "?", "。"].includes(normalized.slice(-1))) {
    normalized = normalized.slice(0, -1).trimEnd();
  }
  return normalized === "你是什么模型";
}

const MODEL_CONTROL_SELECTORS = [
  "#vqa-form button[type='submit']",
  "[data-prompt]",
  "#content-form button[type='submit']",
  "#search-form button[type='submit']",
  "#compare-form button[type='submit']",
  "#analyze",
];

function modelControls() {
  return document.querySelectorAll(MODEL_CONTROL_SELECTORS.join(","));
}

function updateProviderCapabilityGates() {
  const capabilities = state.provider?.capabilities || {};
  const provider = state.provider || {};
  const hardStop = Object.values(provider.errors || {}).some(
    (error) => error?.stop_retries === true,
  );
  const baseUnavailable = provider.mode === "no_model"
    || (provider.mode === "bailian" && !provider.credential?.configured)
    || hardStop;
  const selectorCapabilities = new Map([
    ["#vqa-form button[type='submit']", "vlm"],
    ["[data-prompt]", "vlm"],
    ["#content-form button[type='submit']", "vlm"],
    ["#search-form button[type='submit']", "retrieval"],
    ["#compare-form button[type='submit']", "vlm"],
    ["#analyze", "vlm"],
  ]);
  selectorCapabilities.forEach((capability, selector) => {
    document.querySelectorAll(selector).forEach((control) => {
      const unavailable = baseUnavailable;
      control.dataset.providerDisabled = unavailable ? "true" : "false";
      control.setAttribute("aria-disabled", unavailable ? "true" : "false");
      if (unavailable) control.title = PROVIDER_DISABLED_MESSAGE;
      else if (!capabilities[capability]) control.title = PROVIDER_PREFLIGHT_MESSAGE;
      else if (
        [PROVIDER_DISABLED_MESSAGE, PROVIDER_PREFLIGHT_MESSAGE].includes(control.title)
      ) control.removeAttribute("title");
    });
  });
}

async function ensureProviderForRequest(capability = "vlm", button = null) {
  const provider = await api("/providers/access", {timeoutMs: 8000});
  renderProviderAccess(provider);
  const connection = provider.connection_state || provider.state;
  if (connection === "READY" && provider.capabilities?.[capability]) {
    return provider;
  }
  if (button?.disabled) button.textContent = "正在检查连接";
  showToast(PROVIDER_PREFLIGHT_MESSAGE, 4200);
  const result = await api(
    `/providers/access/preflight?capability=${encodeURIComponent(capability)}`,
    {method: "POST", body: "{}", timeoutMs: 195000},
  );
  renderProviderAccess(result.provider);
  if (!result.success || !result.provider?.capabilities?.[capability]) {
    const errors = Object.values(result.provider?.errors || {});
    const message = errors.find((item) => item?.public_message)?.public_message
      || "当前模型连接仍不可用，按钮已恢复；请稍后重试或切换模型接入方式。";
    throw new Error(message);
  }
  if (button?.disabled) button.textContent = "连接已恢复，正在请求";
  return result.provider;
}

async function modelApi(
  path,
  options = {},
  {capability = "vlm", button = null, activeLabel = "处理中"} = {},
) {
  await ensureProviderForRequest(capability, button);
  if (button?.disabled) button.textContent = activeLabel;
  return api(path, {...options, timeoutMs: options.timeoutMs || 210000});
}

function openModelAccess({firstUse = false} = {}) {
  $("model-access-drawer").classList.remove("hidden");
  $("model-access-entry").setAttribute("aria-expanded", "true");
  if (firstUse) showToast("你可以先选择“暂不接入模型”继续浏览，稍后再配置。", 4200);
}

function closeModelAccess() {
  $("model-access-drawer").classList.add("hidden");
  $("model-access-entry").setAttribute("aria-expanded", "false");
}

function providerStatusText(provider) {
  if (provider.mode === "self_hosted") {
    const scopes = provider.index_scopes || {};
    const product = Number(scopes.product?.items || provider.index?.items || 0);
    const train = Number(scopes.system_train?.items || 0);
    const val = Number(scopes.system_val?.items || 0);
    return `服务器映射 · ${providerStateLabel(provider)} · 当前产品索引 ${product} 项 · 系统索引 Train ${train} / Val ${val}`;
  }
  const vlm = `VLM ${providerCapabilityLabel(provider.vlm?.status, provider.capabilities?.vlm)}`;
  const embedding = `Embedding ${providerCapabilityLabel(provider.embedding?.status, provider.capabilities?.embedding)}`;
  const indexItems = Number(provider.index?.items || 0);
  const indexOnDisk = ["ready", "index_ready_connection_required"].includes(provider.index?.status);
  const index = indexOnDisk
    ? `索引已存在 ${indexItems} 项${provider.capabilities?.retrieval ? "（可用）" : "（连接验证后可用）"}`
    : "索引尚未建立";
  return `${providerModeLabel(provider)} · ${providerStateLabel(provider)} · ${vlm} · ${embedding} · ${index}`;
}

function providerModeLabel(provider) {
  if (provider.mode !== "bailian") return provider.mode_label;
  return provider.cloud_tier === "high_quality" ? "百炼云端高质量" : "百炼云端标准";
}

function providerStateLabel(provider) {
  const connection = String(provider.connection_state || provider.state || "");
  if (connection === "READY") return "READY";
  return {
    NO_MODEL: "未接入模型",
    UNCONFIGURED: "待配置",
    NOT_TESTED: "尚未测试",
    CONNECTING: "正在验证",
    RECONNECTING: "正在重新连接",
    STALE: "验证已过期",
    OFFLINE: "网络离线",
    INVALID_KEY: "API Key 无效",
    BILLING_BLOCKED: "额度或余额不足",
    PERMISSION_DENIED: "模型权限不足",
    REGION_ENDPOINT_MISMATCH: "地域或端点不匹配",
    RATE_LIMITED: "请求限流",
    SERVICE_UNAVAILABLE: "云服务暂不可用",
    NETWORK_TIMEOUT: "网络超时",
    PARTIAL_READY: "部分就绪",
    ERROR: "连接失败",
  }[connection] || "状态待确认";
}

function providerCapabilityLabel(rawStatus, ready) {
  if (ready) return "已连接";
  const status = String(rawStatus || "").toLowerCase();
  if (["not_tested", "not_configured", "connecting"].includes(status)) return "待测试";
  if (["disabled", "unloaded"].includes(status)) return "未启用";
  if (["error", "failed", "unavailable"].includes(status)) return "连接失败";
  return "未就绪";
}

function providerIndexLabel(provider) {
  const status = String(provider.index?.status || "").toLowerCase();
  const items = Number(provider.index?.items || 0);
  if (status === "ready") return `已就绪 · ${items} 项`;
  if (status === "index_ready_connection_required") {
    return `完整索引已存在 · ${items} 项 · 连接验证后可用`;
  }
  if (status === "disabled") return "未启用";
  return items > 0 ? `索引已存在 · ${items} 项` : "尚未建立";
}

function renderProviderAccess(provider) {
  state.provider = provider;
  markRetrievalProjectionPending(provider);
  $("health").textContent = providerStatusText(provider);
  $("model-access-entry").classList.toggle(
    "warn",
    ["NO_MODEL", "PARTIAL_READY", "CONNECTING", "RECONNECTING", "NOT_TESTED", "STALE", "RATE_LIMITED"].includes(
      provider.connection_state || provider.state,
    ),
  );
  $("model-access-entry").classList.toggle(
    "error",
    ["ERROR", "OFFLINE", "INVALID_KEY", "BILLING_BLOCKED", "PERMISSION_DENIED", "REGION_ENDPOINT_MISMATCH", "SERVICE_UNAVAILABLE", "NETWORK_TIMEOUT"].includes(
      provider.connection_state || provider.state,
    ),
  );
  const summaryParts = [
    `当前模式：${provider.mode_label}${provider.mode === "bailian" ? ` · ${providerModeLabel(provider)}` : ""} · ${providerStateLabel(provider)}`,
    `VLM：${provider.vlm?.model_id || "未配置"} · ${providerCapabilityLabel(provider.vlm?.status, provider.capabilities?.vlm)}`,
    `Embedding：${provider.embedding?.model_id || "未配置"}${provider.embedding?.dimensions ? `@${provider.embedding.dimensions}` : ""} · ${providerCapabilityLabel(provider.embedding?.status, provider.capabilities?.embedding)}`,
    `索引：${providerIndexLabel(provider)}`,
    provider.index_availability?.exists && !provider.index_availability?.query_vector_available
      ? "索引文件仍完整保留；当前仅因连接不可用而无法生成查询向量"
      : "索引与连接状态独立",
    Object.keys(provider.errors || {}).length ? "存在需处理的接入错误" : "当前没有接入错误",
  ];
  if (provider.mode === "self_hosted") {
    const scopes = provider.index_scopes || {};
    summaryParts[3] =
      `当前产品索引：${Number(scopes.product?.items || provider.index?.items || 0)} 项；`
      + `系统索引：Train ${Number(scopes.system_train?.items || 0)} / Val ${Number(scopes.system_val?.items || 0)}`;
  }
  $("provider-current-summary").textContent = summaryParts.join("　");
  const modeInput = document.querySelector(`input[name="provider-mode"][value="${provider.mode}"]`);
  if (modeInput) modeInput.checked = true;
  const tierInput = document.querySelector(`input[name="cloud-tier"][value="${provider.cloud_tier || "standard"}"]`);
  if (tierInput) tierInput.checked = true;
  const sourceInput = document.querySelector(`input[name="credential-source"][value="${provider.credential_source || "course_default"}"]`);
  if (sourceInput) sourceInput.checked = true;
  updateProviderSections(provider.mode);
  $("cloud-retrieval-scope").classList.toggle("hidden", provider.mode !== "bailian");
  if (provider.mode === "bailian") {
    const baseItems = Number(provider.index?.base_items || 0);
    const userItems = Number(provider.index?.user_items || 0);
    const scope = `当前完整云端检索索引：${baseItems} 项课程资产 + ${userItems} 项用户增量资产 · 总计 ${baseItems + userItems} 项`;
    $("cloud-index-scope").textContent = scope;
    $("cloud-retrieval-scope").textContent = `${scope}。未入索引图片仍可浏览、选中并直接发送给云端 VLM，但不会出现在检索结果中。`;
    const connectionResult = $("provider-connection-result");
    const courseCredentialMissing = provider.credential_source === "course_default"
      && provider.credential?.configured === false;
    if (courseCredentialMissing) {
      connectionResult.textContent =
        "课程演示默认 API Key 当前不可用；请输入自己的 API Key";
    } else if (provider.capabilities?.vlm && provider.capabilities?.embedding) {
      connectionResult.textContent =
        `已连接 · VLM ${provider.vlm?.model_id || ""} · `
        + `Embedding ${Number(provider.embedding?.dimensions || 2560)} 维`;
    } else if ((provider.connection_state || "") === "STALE") {
      connectionResult.textContent = "历史验证已过期，正在等待重新验证";
    } else if (Object.keys(provider.errors || {}).length) {
      const firstError = Object.values(provider.errors)[0] || {};
      connectionResult.textContent =
        `连接失败${firstError.code ? ` · ${firstError.code}` : ""}`;
    } else {
      connectionResult.textContent = "尚未测试";
    }
  }
  const courseCredentialAvailable =
    provider.credential?.course_default_available !== false;
  $("course-credential-note-title").textContent = courseCredentialAvailable
    ? "课程演示默认 Key 已配置"
    : "课程演示默认 API Key 当前不可用";
  $("course-credential-expiry").textContent = courseCredentialAvailable
    ? "预计于 2026 年 9 月 30 日停用"
    : "默认 Key 当前不可用";
  $("course-credential-note-body").textContent = courseCredentialAvailable
    ? "该默认 Key 仅用于课程演示与课程作业验收；预计在 2026 年 9 月 30 日后由作者手动取消。"
    : "课程演示默认 Key 当前未提供或已停用。这不是网页故障或模型异常；您仍可浏览本地图片、工作区和已有结果。";
  $("course-credential-note-extra").textContent = courseCredentialAvailable
    ? "自 2026 年 10 月 1 日起如无法调用，应按计划内到期停用处理，不是网页故障或模型异常；如需继续使用，请改用自己的阿里云百炼 API Key。API Key 不会通过页面或状态接口回显。"
    : "如需继续使用百炼云端模型，请选择“使用自己的 API Key”并重新测试连接。";
  if (provider.mode === "self_hosted") {
    const scopes = provider.index_scopes || {};
    $("self-hosted-index-scope").innerHTML = [
      `<strong>产品/自定义资产：${Number(scopes.product?.active_unique_images ?? scopes.product?.items ?? 0)} 张可检索唯一图片</strong>`,
      `<p>活动记录：${Number(scopes.product?.active_records || 0)} 条 · 历史封存：${Number(scopes.product?.archived_records || 0)} 条</p>`,
      `<details><summary>高级诊断</summary><p>Faiss 物理向量：${Number(scopes.product?.physical_vectors || 0)} · 全部唯一 SHA：${Number(scopes.product?.total_unique_sha || 0)} · 重复 SHA 记录：${Number(scopes.product?.duplicate_sha_records || 0)}</p></details>`,
      `<p>系统 Train 索引：${Number(scopes.system_train?.items || 0)} 项 · ${escapeHtml(scopes.system_train?.status || "待确认")}</p>`,
      `<p>系统 Val 索引：${Number(scopes.system_val?.items || 0)} 项 · ${escapeHtml(scopes.system_val?.status || "待确认")}</p>`,
      "<p>当前图库、Train、Val 与全部图库按各自索引检索；物理向量数不等于当前可搜索图片数。</p>",
    ].join("");
  }
  updateCredentialPanels();
  updateProviderCapabilityGates();
}

function markRetrievalProjectionPending(provider) {
  const badge = $("retriever-badge");
  const meta = $("retriever-meta");
  if (!badge || !meta) return;
  const mode = provider?.mode || "no_model";
  const canRetrieve = Boolean(provider?.capabilities?.retrieval);
  badge.classList.remove("warn", "ready");
  badge.dataset.providerMode = mode;
  badge.dataset.runtimeBackend = "";
  badge.dataset.indexIdentity = "";
  badge.dataset.dimension = "";
  badge.dataset.fallbackActive = "false";
  if (mode === "no_model") {
    badge.textContent = "检索暂不可用";
    meta.textContent = "当前未接入模型；已有图片与结果仍可浏览。";
    return;
  }
  badge.textContent = canRetrieve ? "正在同步检索状态" : "等待检索连接";
  meta.textContent = canRetrieve
    ? "正在核对当前 Provider、查询编码器与活动索引。"
    : "当前 Provider 尚未形成可执行的检索链路。";
}

function retrievalRuntimeProjection(status) {
  const retrieval = status?.retrieval || {};
  const provider = status?.provider || state.provider || {};
  // Historical candidate expression:
  // retrieval.active_backend || retrieval.retrieval_backend
  // The requested backend is now diagnostic-only; an inactive preserved
  // index must not be presented as the currently executable backend.
  const activeBackend = String(retrieval.active_backend || "");
  const requestedBackend = String(
    retrieval.retrieval_backend
      || retrieval.requested_backend
      || status?.default_retriever
      || "",
  );
  const backend = activeBackend;
  const runtimeStatus = String(retrieval.status || "").toLowerCase();
  const dimension = Number(
    retrieval.dimensions
      || retrieval.embedding?.dimensions
      || retrieval.index?.dimensions
      || provider.embedding?.dimensions
      || 0,
  );
  const indexIdentity = String(
    retrieval.index_version || retrieval.index?.index_version || provider.index?.index_version || "",
  );
  const fallback = Boolean(retrieval.fallback_active);
  const mode = provider.mode || state.provider?.mode || "no_model";
  if (fallback || backend === "r0") {
    return {
      label: "基础检索模式",
      meta: `当前真实使用基础回退检索${retrieval.fallback_reason ? ` · ${retrieval.fallback_reason}` : ""}`,
      tone: "warn",
      backend: backend || "r0",
      dimension,
      indexIdentity: indexIdentity || "r0-color-grid-v1",
      fallback: true,
    };
  }
  if (
    !activeBackend
    && (
      runtimeStatus === "index_ready_connection_required"
      || runtimeStatus === "not_configured"
      || runtimeStatus === "not_tested"
      || runtimeStatus === "connecting"
      || runtimeStatus === "unavailable"
    )
  ) {
    return {
      label: mode === "no_model" ? "检索暂不可用" : "等待检索连接",
      meta: indexIdentity
        ? `索引 ${indexIdentity} 已保留；当前 ${requestedBackend || "查询编码器"} 尚未形成可执行检索链路。`
        : "当前 Provider 未验证或检索索引尚未绑定。",
      tone: "idle",
      backend: "",
      dimension,
      indexIdentity,
      fallback: false,
    };
  }
  if (backend === "bailian_cloud_e1" && dimension === 2560 && indexIdentity) {
    return {
      label: "CLOUD E1 · 2560D",
      meta: `云端向量检索 · ${Number(retrieval.items || 0)} 项 · ${indexIdentity}`,
      tone: "ready",
      backend,
      dimension,
      indexIdentity,
      fallback: false,
    };
  }
  if (backend === "e1" && mode === "self_hosted" && dimension === 2048 && indexIdentity) {
    return {
      label: "SELF-HOSTED E1 · 2048D",
      meta: `服务器向量检索 · ${Number(retrieval.items || 0)} 项 · ${indexIdentity}`,
      tone: "ready",
      backend,
      dimension,
      indexIdentity,
      fallback: false,
    };
  }
  if (backend === "e1" && mode === "local" && dimension && indexIdentity) {
    return {
      label: `LOCAL E1 · ${dimension}D`,
      meta: `本地向量检索 · ${Number(retrieval.items || 0)} 项 · ${indexIdentity}`,
      tone: "ready",
      backend,
      dimension,
      indexIdentity,
      fallback: false,
    };
  }
  if (backend === "disabled" || mode === "no_model") {
    return {
      label: "检索暂不可用",
      meta: "当前未接入可执行的检索模型。",
      tone: "idle",
      backend: backend || "disabled",
      dimension,
      indexIdentity,
      fallback: false,
    };
  }
  return {
    label: "等待检索连接",
    meta: "当前 Provider 未验证或检索索引尚未绑定。",
    tone: "idle",
    backend: backend || "not_ready",
    dimension,
    indexIdentity,
    fallback: false,
  };
}

function renderRetrievalRuntimeStatus(status) {
  const projection = retrievalRuntimeProjection(status);
  state.retrievalRuntime = {status, projection};
  const badge = $("retriever-badge");
  badge.textContent = projection.label;
  badge.classList.toggle("warn", projection.tone === "warn");
  badge.classList.toggle("ready", projection.tone === "ready");
  badge.dataset.providerMode = status?.provider?.mode || state.provider?.mode || "";
  badge.dataset.runtimeBackend = projection.backend;
  badge.dataset.indexIdentity = projection.indexIdentity;
  badge.dataset.dimension = String(projection.dimension || "");
  badge.dataset.fallbackActive = String(projection.fallback);
  badge.title = projection.meta;
  badge.setAttribute("aria-label", `${projection.label}；${projection.meta}`);
  $("retriever-meta").textContent = projection.meta;
  return projection;
}

function updateProviderSections(mode = null) {
  const selected = mode
    || document.querySelector('input[name="provider-mode"]:checked')?.value
    || "no_model";
  $("cloud-tier-section").classList.toggle("hidden", selected !== "bailian");
  $("local-provider-section").classList.toggle("hidden", selected !== "local");
  $("self-hosted-provider-section").classList.toggle("hidden", selected !== "self_hosted");
}

async function loadCloudIndexStatus() {
  const status = await api("/providers/access/cloud-index");
  $("cloud-index-scope").textContent = status.scope_message;
  const full = status.full_course_coverage || {};
  $("cloud-full-index-scope").textContent =
    `课程全量云索引：Train ${Number(full.train_completed || 0)}/${Number(full.train_total || 2000)} · `
    + `Val ${Number(full.val_completed || 0)}/${Number(full.val_total || 369)} · `
    + `总计 ${Number(full.total_completed || 0)}/${Number(full.total || 2369)} · `
    + `${Number(full.faiss_ntotal || 0)} 条向量 · 云端标准与云端高质量共用`;
  $("cloud-retrieval-scope").textContent = `${status.scope_message}。未入索引图片仍可浏览、选中并直接发送给云端 VLM，但不会出现在检索结果中。`;
  $("cloud-index-result").textContent = status.status === "ready"
    ? `已就绪 · ${Number(status.items || 0)} 项 · ${Number(status.dimensions || 2560)} 维`
    : status.status === "index_ready_connection_required"
      ? `完整索引已存在 · ${Number(status.items || 0)} 项 · 连接验证后可用`
      : "尚未建立";
  const fullReady = full.status === "ready"
    && Number(full.total_completed || 0) === 2369
    && Number(full.faiss_ntotal || 0) === 2369;
  $("cloud-index-build-first10").disabled = fullReady;
  $("cloud-index-build-first10").textContent = fullReady
    ? "完整云索引已建立"
    : "检查云索引状态";
  $("cloud-index-build-first10").title = fullReady
    ? "2369 项完整索引已注册；旧的 Train 前十建立动作已停用。"
    : "当前没有完整索引，仅用于检查历史索引状态。";
  return status;
}

function selectedCredentialSource() {
  return document.querySelector('input[name="credential-source"]:checked')?.value
    || "course_default";
}

function updateCredentialPanels() {
  const own = selectedCredentialSource() === "user_session";
  $("user-credential-fields").classList.toggle("hidden", !own);
  $("course-credential-note").classList.toggle("hidden", own);
}

async function saveUserCredentialIfNeeded({required = false} = {}) {
  if (selectedCredentialSource() !== "user_session") return;
  const apiKey = $("provider-api-key").value;
  if (!apiKey) {
    if (required && !state.provider?.credential?.configured) {
      throw new Error("请先输入自己的 API Key。");
    }
    return;
  }
  await api("/providers/access/credentials", {
    method: "POST",
    body: JSON.stringify({
      api_key: apiKey,
      region: $("provider-region").value,
      api_host: $("provider-api-host").value || null,
      endpoint_mode: $("provider-endpoint-mode").value,
      workspace_id: $("provider-workspace-id").value || null,
      only_this_session: true,
    }),
  });
  $("provider-api-key-summary").textContent =
    `已安全保存：${apiKey.slice(0, 3)}****${apiKey.slice(-4)} · 当前后端运行会话`;
  $("provider-api-key").value = "";
}

async function loadProviderAccess() {
  const [provider, profiles] = await Promise.all([
    api("/providers/access"),
    state.providerProfiles ? Promise.resolve(state.providerProfiles) : api("/providers/profiles"),
  ]);
  state.providerProfiles = profiles;
  renderProviderAccess(provider);
  await loadCloudIndexStatus();
  if (provider.selection_required) openModelAccess({firstUse: true});
  return provider;
}

document.addEventListener("click", (event) => {
  const control = event.target.closest?.("[data-provider-disabled='true']");
  if (!control) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  showToast(PROVIDER_DISABLED_MESSAGE, 4600);
}, true);

document.addEventListener("submit", (event) => {
  const submitter = event.submitter;
  if (submitter?.dataset.providerDisabled !== "true") return;
  if (
    event.target?.id === "vqa-form"
    && isExactRuntimeModelIdentityQuestion($("vqa-question").value)
  ) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  showToast(PROVIDER_DISABLED_MESSAGE, 4600);
}, true);

function setBusy(button, busy, label = "处理中") {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.label || button.textContent;
    button.disabled = false;
  }
}

function recordTrace(payload) {
  const requestId = payload?.request_id || payload?.response?.request_id;
  if (requestId) $("trace-result").textContent = requestId;
}

function normalizedLibraryAsset(item) {
  return {
    ...item,
    ref: `library:${item.image_id}`,
    source: "library",
    sourceAssetId: item.image_id,
    asset_id: item.image_id,
    image_url: item.image_url || `/library/${encodeURIComponent(item.image_id)}/image`,
  };
}

function normalizedLocalAsset(item) {
  return {
    ...item,
    ref: `local:${item.asset_id}`,
    source: "local",
    sourceAssetId: item.asset_id,
    image_url: item.image_url || `/local-assets/${item.asset_id}/image`,
  };
}

function normalizedSystemAsset(item) {
  return {
    ...item,
    ref: `system:${item.asset_id}`,
    source: "system",
    sourceAssetId: item.asset_id,
    image_id: item.original_filename || item.image_id || item.asset_id,
    image_url: item.image_url || `/visual-assets/${encodeURIComponent(item.asset_id)}/image`,
  };
}

function normalizedSessionAsset(item) {
  return {
    ...item,
    ref: `session:${item.asset_id}`,
    source: "session",
    sourceAssetId: item.asset_id,
    conversationId: item.conversation_id,
    image_url: item.image_url,
  };
}

function normalizedPublicCourseAsset(item) {
  const source = item.source || "library";
  const sourceAssetId = item.source_asset_id || item.sourceAssetId
    || String(item.asset_id || "").replace(/^local:/, "");
  const base = {
    ...item,
    source,
    sourceAssetId,
    ref: item.ref || `${source}:${sourceAssetId}`,
    image_id: item.image_id || sourceAssetId,
    image_url: item.image_url,
    thumbnail_url: item.thumbnail_url || item.image_url,
  };
  if (source === "system") return normalizedSystemAsset({...base, asset_id: sourceAssetId});
  if (source === "local") return normalizedLocalAsset({...base, asset_id: sourceAssetId});
  if (source === "session") {
    return normalizedSessionAsset({
      ...base,
      asset_id: sourceAssetId,
      conversation_id: item.conversation_id || state.currentConversationId,
    });
  }
  return normalizedLibraryAsset({...base, image_id: sourceAssetId});
}

function absorbAssets(items = []) {
  const normalized = items.map(normalizedPublicCourseAsset);
  state.assets = [
    ...new Map(
      [...state.assets, ...normalized].map((item) => [item.ref, item]),
    ).values(),
  ];
  return normalized;
}

function operationId(prefix = "op") {
  if (window.crypto?.randomUUID) return `${prefix}-${window.crypto.randomUUID()}`;
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function assetRequest(asset) {
  return {
    source: asset.source,
    asset_id: asset.sourceAssetId,
    ...(asset.source === "session"
      ? {conversation_id: asset.conversationId || state.currentConversationId}
      : {}),
  };
}

function findAsset(ref) {
  return state.assets.find((item) => item.ref === ref) || null;
}

function activeAssets() {
  return state.contextRefs.map(findAsset).filter(Boolean);
}

function workspaceAssets(kind) {
  return (state.workspaces[kind]?.refs || []).map(findAsset).filter(Boolean);
}

function currentWorkspaceKind() {
  return {
    generate: "generation",
    retrieve: "retrieval",
    rank: "compare",
  }[state.currentView] || (state.currentView === "chat" ? "chat" : null);
}

function projectedSelectionRefs() {
  const kind = currentWorkspaceKind();
  if (kind === "chat") return new Set(state.contextRefs);
  if (kind && state.workspaces[kind]) {
    return new Set(state.workspaces[kind].refs);
  }
  return new Set();
}

function currentBatchSelection() {
  const kind = currentWorkspaceKind();
  return state.batchSelections[kind] || new Set();
}

function orderedBatchAssets() {
  return [...currentBatchSelection()].map(findAsset).filter(Boolean);
}

function syncAssetListState() {
  const kind = currentWorkspaceKind();
  const members = projectedSelectionRefs();
  const selected = currentBatchSelection();
  document.querySelectorAll(".asset-item[data-asset-ref]").forEach((node) => {
    const ref = node.dataset.assetRef;
    const pending = state.chatPendingRefs.has(ref)
      || Object.values(state.workspaces).some(
        (workspace) => workspace.pendingRefs.has(ref),
      );
    node.classList.toggle("active", state.selectedRef === ref);
    node.classList.toggle("batch-selected", selected.has(ref));
    node.classList.toggle("in-context", members.has(ref));
    node.classList.toggle("pending", pending);
    const checkbox = node.querySelector(".asset-projection-check");
    if (checkbox) checkbox.checked = selected.has(ref);
    const status = node.querySelector(".asset-member-state");
    if (status) status.textContent = members.has(ref) ? "已加入" : "未加入";
  });
  if ($("batch-selection-summary")) {
    if (!kind) {
      $("batch-selection-summary").textContent =
        "当前页面不使用批量选图；左侧显示当前 Provider 索引状态。";
      $("batch-selection-toolbar").classList.add("hidden");
      return;
    }
    const target = {
      chat: "Chat 当前对话",
      generation: "生成工作区",
      retrieval: "检索工作区",
      compare: "比较工作区",
    }[kind];
    $("batch-selection-summary").textContent =
      `已选择 ${selected.size} 张 · 目标：${target}`;
    $("batch-selection-target").textContent = `将加入：${target}`;
    $("batch-selection-toolbar").classList.toggle("hidden", selected.size === 0);
  }
}

function applyWorkspaceSnapshot(kind, snapshot) {
  const workspace = state.workspaces[kind];
  if (!workspace || !snapshot) return;
  const normalized = absorbAssets(snapshot.selected_assets || []);
  workspace.workspace_id = snapshot.workspace_id || workspace.workspace_id;
  workspace.version = Number(snapshot.version || 0);
  workspace.refs = normalized.map((item) => item.ref);
  applyWorkspaceOptions(kind, snapshot.local_options || {});
  renderFunctionWorkspaces();
  renderAssetList();
}

async function refreshWorkspaceForView(view) {
  const kind = {generate: "generation", retrieve: "retrieval", rank: "compare"}[view];
  if (!kind) {
    renderAssetList();
    return;
  }
  const workspace = state.workspaces[kind];
  workspace.loadSequence = Number(workspace.loadSequence || 0) + 1;
  const sequence = workspace.loadSequence;
  try {
    const snapshot = await api(`/course/workspaces/${kind}`);
    if (sequence !== workspace.loadSequence || state.currentView !== view) return;
    applyWorkspaceSnapshot(kind, snapshot);
  } catch (error) {
    showError(error.message);
  }
}

const viewAliases = {
  chat: "chat",
  generate: "generate",
  generation: "generate",
  retrieve: "retrieve",
  retrieval: "retrieve",
  rank: "rank",
  ranking: "rank",
  library: "library",
  history: "history",
};

function navigate(view, {syncHash = true} = {}) {
  clearToast();
  view = viewAliases[view] || "chat";
  state.currentView = view;
  document.querySelectorAll(".nav-item").forEach((node) => node.classList.toggle("active", node.dataset.view === view));
  document.querySelectorAll(".view").forEach((node) => node.classList.toggle("active", node.id === `view-${view}`));
  $("chat-context-bar").classList.toggle("context-hidden", view !== "chat");
  const addButton = $("add-context");
  if (addButton) {
    addButton.textContent = {
      chat: "加入 Chat",
      generate: "加入生成工作区",
      retrieve: "加入检索工作区",
      rank: "加入比较工作区",
    }[view] || "查看图片";
    addButton.disabled = !["chat", "generate", "retrieve", "rank"].includes(view);
  }
  if (syncHash && window.location.hash !== `#${view}`) window.location.hash = view;
  if (view === "history") loadHistory();
  refreshWorkspaceForView(view);
  // Route changes must re-project the actual active retrieval backend instead
  // of carrying a badge rendered for a previous view/provider snapshot.
  void loadStatus({renderSystemCards: false}).catch(() => null);
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.onclick = () => navigate(button.dataset.view);
});
window.addEventListener("hashchange", () => navigate(window.location.hash.slice(1), {syncHash: false}));
navigate(window.location.hash.slice(1) || "chat", {syncHash: false});

function providerIndexStateMarkup(item) {
  if (item.source !== "local") return "";
  const status = item.index_lifecycle?.active_provider || {
    status: "waiting_for_provider",
    label: "等待模型",
  };
  const icons = {
    indexed: "✓",
    pending: "!",
    failed: "×",
    indexing: "↻",
    identity_mismatch: "↻",
    waiting_for_provider: "—",
  };
  const title = `${status.provider_label || "当前 Provider"}：${status.label}`;
  return `<button type="button" class="asset-index-state ${escapeHtml(
    String(status.status || "").replaceAll("_", "-"),
  )}" data-index-asset="${escapeHtml(item.sourceAssetId)}" title="${escapeHtml(
    title,
  )}" aria-label="${escapeHtml(title)}"><span class="index-state-icon" aria-hidden="true">${
    icons[status.status] || "—"
  }</span><span>${escapeHtml(status.label || "等待模型")}</span></button>`;
}

function renderAssetList() {
  const context = projectedSelectionRefs();
  const batchSelectable = Boolean(currentWorkspaceKind());
  const lifecycleText = (item) => {
    const lifecycle = item.index_lifecycle;
    if (!lifecycle) return "";
    const canonical = lifecycle.canonical?.status === "completed" ? "Canonical 已完成" : "Canonical 待标注";
    const local = lifecycle.local_e1?.status === "completed" ? "本地 E1 已完成" : "本地 E1 未索引";
    const cloud = lifecycle.cloud_e1?.status === "completed" ? "云 E1 已完成" : "云 E1 未索引";
    return ` · ${canonical} · ${local} · ${cloud}`;
  };
  const categoryStrip = (item) => {
    const categories = Array.isArray(item.two_layer?.categories)
      ? item.two_layer.categories.slice(0, 3)
      : [];
    return categories.length
      ? `<span class="asset-category-strip">${categories.map(
        (tag) => `<span>${escapeHtml(tag)}</span>`,
      ).join("")}</span>`
      : "";
  };
  const render = (items) => items
    .map((item) => `
      <div class="asset-item ${state.selectedRef === item.ref ? "active" : ""} ${context.has(item.ref) ? "in-context" : ""} ${
        state.chatPendingRefs.has(item.ref)
        || Object.values(state.workspaces).some((workspace) => workspace.pendingRefs.has(item.ref))
          ? "pending"
          : ""
      }" data-asset-ref="${escapeHtml(item.ref)}">
        ${batchSelectable
          ? `<input class="asset-projection-check" type="checkbox" aria-label="批量选择 ${escapeHtml(item.image_id)}" ${currentBatchSelection().has(item.ref) ? "checked" : ""}>`
          : `<span class="asset-selection-placeholder" aria-hidden="true"></span>`}
        <button type="button" class="asset-item-main" aria-label="查看 ${escapeHtml(item.image_id)}">
          <img loading="lazy" decoding="async" src="${escapeHtml(item.thumbnail_url || item.image_url)}" alt="">
          <span><strong>${escapeHtml(item.image_id)}</strong><small>${
          item.source === "system"
            ? `${item.width || "?"} × ${item.height || "?"} · ${item.source_split === "train" ? "训练库" : "验证库"} · ${item.label_status === "pending" ? "待标注" : "机器暂定"}`
            : item.source === "library"
              ? `${item.width || "?"} × ${item.height || "?"} · 历史示例`
              : item.source === "session"
                ? "临时 · 当前会话 · 不写入图库"
                : `${
                  item.two_layer?.default?.["主题"] || "自定义图库"
                }${lifecycleText(item) || ` · ${item.label_status === "machine_provisional" ? "机器暂定" : "待标注"}`}`
        }</small>${categoryStrip(item)}</span>
        </button>
        <span class="asset-row-status">
          <span class="asset-member-state">${context.has(item.ref) ? "已加入" : "未加入"}</span>
          ${providerIndexStateMarkup(item)}
        </span>
      </div>`).join("");
  $("library").innerHTML = render(state.libraryAssets) || '<p class="empty-context">没有匹配资产</p>';
  $("local-assets").innerHTML = render(state.localAssets) || '<p class="empty-context">当前库暂无自定义图片</p>';
  $("session-assets").innerHTML = render(state.sessionAssets) || '<p class="empty-context">当前页面会话暂无临时图片</p>';
  $("library-count").textContent = state.libraryTotal || state.libraryAssets.length;
  $("local-count").textContent = state.localAssets.length;
  $("session-count").textContent = state.sessionAssets.length;
  document.querySelectorAll(".asset-item[data-asset-ref]").forEach((node) => {
    const image = node.querySelector("img");
    if (image) {
      image.onerror = () => {
        node.classList.add("thumbnail-error");
        image.removeAttribute("src");
      };
    }
  });
  syncAssetListState();
}

for (const containerId of ["library", "local-assets", "session-assets"]) {
  const container = $(containerId);
  container.addEventListener("change", (event) => {
    const checkbox = event.target.closest(".asset-projection-check");
    if (!checkbox) return;
    const row = checkbox.closest("[data-asset-ref]");
    const selected = currentBatchSelection();
    if (checkbox.checked) selected.add(row.dataset.assetRef);
    else selected.delete(row.dataset.assetRef);
    syncAssetListState();
  });
  container.addEventListener("click", (event) => {
    if (event.target.closest(".asset-projection-check")) return;
    const main = event.target.closest(".asset-item-main");
    if (!main || event.detail > 1) return;
    const row = main.closest("[data-asset-ref]");
    selectAsset(row.dataset.assetRef);
  });
  container.addEventListener("dblclick", (event) => {
    if (event.target.closest(".asset-projection-check")) return;
    const row = event.target.closest("[data-asset-ref]");
    if (!row) return;
    event.preventDefault();
    addAssetToCurrentWorkspace(row.dataset.assetRef);
  });
}

function renderContext() {
  $("context-count").textContent = activeAssets().length;
  renderChatAssetState();
  renderFunctionWorkspaces();
  renderAssetList();
}

function detectedSearchMode() {
  const hasText = Boolean($("search-query")?.value.trim());
  const hasImages = workspaceAssets("retrieval").length > 0;
  if (hasText && hasImages) return "hybrid";
  if (hasImages) return "image";
  if (hasText) return "text";
  return null;
}

function updateDetectedSearchMode() {
  const target = $("search-detected-mode");
  if (!target) return;
  const labels = {
    text: "已识别：文字检索",
    image: "已识别：以图搜图",
    hybrid: "已识别：图文联合检索",
  };
  const mode = detectedSearchMode();
  target.textContent = labels[mode] || "当前：请输入文字或加入查询图片";
  target.dataset.mode = mode || "empty";
}

function updateCompareControls() {
  const selectMode = $("compare-action")?.value === "select";
  const field = $("compare-select-count-field");
  if (field) field.classList.toggle("hidden", !selectMode);
  if ($("compare-select-count")) {
    $("compare-select-count").disabled = !selectMode;
  }
}

function workspaceLocalOptions(kind) {
  if (kind === "generation") {
    return {
      content_type: $("content-mode")?.value || "auto",
      target_length: state.contentLengthUserEdited
        ? Number($("content-length")?.value || 0)
        : null,
      target_length_user_edited: state.contentLengthUserEdited,
      organization: $("content-organization")?.value || "input_order",
      natural_language_request: $("content-request")?.value || "",
    };
  }
  if (kind === "retrieval") {
    return {
      detected_mode: detectedSearchMode(),
      text_query: $("search-query")?.value || "",
      top_k: Number($("search-top-k")?.value || 5),
      exclude_query_images: Boolean($("search-exclude-query")?.checked),
      library_scope: $("search-library-scope")?.value || "all_libraries",
      current_library_id: $("library-select")?.value || null,
    };
  }
  return {
    action: $("compare-action")?.value || "select",
    criterion: $("compare-instruction")?.value || "",
    select_count: Number($("compare-select-count")?.value || 1),
  };
}

async function persistWorkspace(kind, lastResult = null) {
  return enqueueWorkspaceOperation(kind, "replace", {
    selectedAssets: workspaceAssets(kind),
    localOptions: workspaceLocalOptions(kind),
    lastResult,
  });
}

function enqueueWorkspaceOperation(kind, action, {
  asset = null,
  assetRef = null,
  selectedAssets = null,
  direction = 0,
  localOptions = null,
  lastResult = null,
} = {}) {
  const workspace = state.workspaces[kind];
  workspace.loadSequence = Number(workspace.loadSequence || 0) + 1;
  const opId = operationId(kind);
  const pendingRef = asset?.ref || assetRef;
  if (pendingRef) workspace.pendingRefs.add(pendingRef);
  renderAssetList();
  const execute = async () => {
    workspace.clientSequence += 1;
    const body = {
      operation_id: opId,
      workspace_id: workspace.workspace_id,
      action,
      client_sequence: workspace.clientSequence,
      expected_version: workspace.version,
      direction,
    };
    if (asset) body.asset = assetRequest(asset);
    if (assetRef) body.asset_ref = assetRef;
    if (selectedAssets) {
      body.selected_assets = selectedAssets.map(assetRequest);
    }
    if (localOptions !== null) body.local_options = localOptions;
    if (lastResult !== null) body.last_result = lastResult;
    let snapshot;
    try {
      snapshot = await api(`/course/workspaces/${kind}/operations`, {
        method: "POST",
        body: JSON.stringify(body),
      });
    } catch (error) {
      if (error.status !== 409) throw error;
      const current = await api(`/course/workspaces/${kind}`);
      applyWorkspaceSnapshot(kind, current);
      body.workspace_id = workspace.workspace_id;
      body.expected_version = workspace.version;
      snapshot = await api(`/course/workspaces/${kind}/operations`, {
        method: "POST",
        body: JSON.stringify(body),
      });
    }
    applyWorkspaceSnapshot(kind, snapshot);
    return snapshot;
  };
  const pending = workspace.queue.then(execute, execute);
  workspace.queue = pending.catch(() => {});
  return pending.finally(() => {
    if (pendingRef) workspace.pendingRefs.delete(pendingRef);
    renderAssetList();
  });
}

function workspaceResultSnapshot(kind, payload) {
  if (kind === "generation") {
    return {
      workspace_result_kind: kind,
      display_text: payload.display_text,
      final_text: payload.final_text,
      content: payload.content,
      options: payload.options,
      resolved_options: payload.resolved_options,
      prompt_candidate: payload.prompt_candidate,
      product_contract_valid: payload.product_contract_valid,
      length_contract: payload.length_contract,
      image_coverage: payload.image_coverage,
      repair_applied: payload.repair_applied,
      risk_sanitized: payload.risk_sanitized,
      fallback_applied: payload.fallback_applied,
      fallback_source: payload.fallback_source,
      contract_errors: payload.contract_errors,
      story_structure: payload.story_structure,
      request_id: payload.request_id,
      result: payload.result ? {
        model: payload.result.model,
        model_revision: payload.result.model_revision,
      } : null,
    };
  }
  if (kind === "retrieval") {
    return {
      workspace_result_kind: kind,
      mode: payload.mode,
      results: payload.results,
      baseline_label: payload.baseline_label,
      index_version: payload.index_version,
      fallback_used: payload.fallback_used,
      fallback_reason: payload.fallback_reason,
      request_id: payload.request_id,
    };
  }
  return {
    workspace_result_kind: kind,
    display_text: payload.display_text,
    public_answer: payload.public_answer,
    ranking: payload.ranking,
    selected: payload.selected,
    action: payload.action,
    repair_level: payload.repair_level,
    model_contract_errors: payload.model_contract_errors,
    prompt_candidate: payload.prompt_candidate,
    product_contract_valid: payload.product_contract_valid,
    request_id: payload.request_id,
  };
}

function applyWorkspaceOptions(kind, options = {}) {
  if (kind === "generation") {
    if (options.content_type) $("content-mode").value = options.content_type;
    applyContentLengthProfile({resetValue: true});
    if (Number(options.target_length) > 0) {
      $("content-length").value = Number(options.target_length);
    }
    state.contentLengthUserEdited = Boolean(
      options.target_length_user_edited,
    );
    if (options.organization) {
      $("content-organization").value = options.organization;
    }
    if (options.natural_language_request !== undefined) {
      $("content-request").value = options.natural_language_request;
    }
    state.contentTypeUserSelected = Boolean(
      options.content_type && options.content_type !== "auto",
    );
    return;
  }
  if (kind === "retrieval") {
    $("search-query").value = options.text_query || options.query_text || "";
    if (Number(options.top_k) > 0) {
      $("search-top-k").value = Number(options.top_k);
    }
    if (options.exclude_query_images !== undefined) {
      $("search-exclude-query").checked = Boolean(
        options.exclude_query_images,
      );
    }
    if (
      options.library_scope
      && [...$("search-library-scope").options].some(
        (option) => option.value === options.library_scope,
      )
    ) {
      $("search-library-scope").value = options.library_scope;
    }
    updateDetectedSearchMode();
    return;
  }
  if (options.action && options.action !== "compare") {
    $("compare-action").value = options.action;
  }
  if (options.criterion !== undefined) {
    $("compare-instruction").value = options.criterion;
  }
  if (Number(options.select_count) > 0) {
    $("compare-select-count").value = Number(options.select_count);
  }
  updateCompareControls();
}

function restoreWorkspaceResult(kind, result) {
  if (!result || result.workspace_result_kind !== kind) return;
  if (kind === "generation") renderStructuredGeneration(result);
  if (kind === "retrieval" && Array.isArray(result.results)) {
    renderSearchResults(result);
  }
  if (kind === "compare" && Array.isArray(result.ranking)) {
    renderRanking(result);
  }
}

function renderFunctionWorkspaces() {
  for (const kind of ["generation", "retrieval", "compare"]) {
    const assets = workspaceAssets(kind);
    const count = $(`${kind}-workspace-count`);
    const target = $(`${kind}-workspace-assets`);
    if (count) count.textContent = assets.length;
    if (!target) continue;
    target.innerHTML = assets.length
      ? assets.map((item, index) => `
          <div class="workspace-chip">
            <img loading="lazy" decoding="async" src="${escapeHtml(item.thumbnail_url || item.image_url)}" alt="">
            <span>图片 ${index + 1}</span>
            <button type="button" data-workspace-move="${kind}" data-workspace-ref="${escapeHtml(item.ref)}" data-direction="-1" aria-label="前移">←</button>
            <button type="button" data-workspace-move="${kind}" data-workspace-ref="${escapeHtml(item.ref)}" data-direction="1" aria-label="后移">→</button>
            <button type="button" data-workspace-remove="${kind}" data-workspace-ref="${escapeHtml(item.ref)}" aria-label="移除">×</button>
          </div>`).join("")
      : `<span class="empty-context">${kind === "retrieval" ? "文本检索可保持为空" : "尚未选择图片"}</span>`;
  }
  const compareAssets = workspaceAssets("compare");
  $("compare-picker").innerHTML = compareAssets.length
    ? compareAssets.map((item, index) => `<div class="rank-asset"><img loading="lazy" decoding="async" src="${escapeHtml(item.thumbnail_url || item.image_url)}" alt=""><span>图片 ${index + 1}</span></div>`).join("")
    : '<span class="empty-context">请在比较工作区加入 2-5 张图片</span>';
  const retrievalAssets = workspaceAssets("retrieval");
  $("query-asset-note").textContent = retrievalAssets.length
    ? `将使用本工作区的 ${retrievalAssets.length} 张图片作为查询条件。`
    : "没有查询图片时，只需输入文字即可检索。";
  updateDetectedSearchMode();
  document.querySelectorAll("[data-workspace-remove]").forEach((button) => {
    button.onclick = () => {
      const kind = button.dataset.workspaceRemove;
      enqueueWorkspaceOperation(kind, "remove", {
        assetRef: button.dataset.workspaceRef,
        localOptions: workspaceLocalOptions(kind),
      }).catch((error) => showError(error.message));
    };
  });
  document.querySelectorAll("[data-workspace-move]").forEach((button) => {
    button.onclick = () => {
      const kind = button.dataset.workspaceMove;
      enqueueWorkspaceOperation(kind, "move", {
        assetRef: button.dataset.workspaceRef,
        direction: Number(button.dataset.direction),
        localOptions: workspaceLocalOptions(kind),
      }).catch((error) => showError(error.message));
    };
  });
}

async function addSelectedToWorkspace(kind) {
  const assets = orderedBatchAssets();
  if (!assets.length) return showError("请先勾选要加入当前工作区的图片。");
  const snapshot = await enqueueWorkspaceOperation(kind, "add_many", {
    selectedAssets: assets,
    localOptions: workspaceLocalOptions(kind),
  });
  state.batchSelections[kind].clear();
  renderAssetList();
  showToast(`已加入${assets.length}张图片，批量选择已清空。`);
  return snapshot;
}

async function importChatToWorkspace(kind) {
  const assets = activeChatBindings()
    .map((binding) => findAsset(binding.ref))
    .filter(Boolean)
    .slice(0, 5);
  if (!assets.length) return showError("当前 Chat 没有可导入图片。");
  return enqueueWorkspaceOperation(kind, "add_many", {
    selectedAssets: assets,
    localOptions: workspaceLocalOptions(kind),
  });
}

for (const kind of ["generation", "retrieval", "compare"]) {
  $(`${kind}-add-selected`).onclick = () => addSelectedToWorkspace(kind).catch((error) => showError(error.message));
  $(`${kind}-import-chat`).onclick = () => importChatToWorkspace(kind).catch((error) => showError(error.message));
  $(`${kind}-clear`).onclick = () => enqueueWorkspaceOperation(
    kind,
    "clear",
    {localOptions: workspaceLocalOptions(kind)},
  ).catch((error) => showError(error.message));
}

$("batch-add-selected").onclick = async () => {
  const kind = currentWorkspaceKind() || "chat";
  try {
    if (kind === "chat") await addBatchToChat();
    else await addSelectedToWorkspace(kind);
  } catch (error) {
    showError(error.message);
  }
};
$("batch-clear-selected").onclick = () => {
  currentBatchSelection().clear();
  renderAssetList();
  showToast("已取消当前工作区的批量选择。");
};

async function loadFunctionWorkspaces() {
  const payload = await api("/course/workspaces");
  for (const item of payload.items || []) {
    const kind = item.workspace_kind;
    if (!state.workspaces[kind]) continue;
    applyWorkspaceSnapshot(kind, item);
  }
  renderFunctionWorkspaces();
  for (const item of payload.items || []) {
    restoreWorkspaceResult(item.workspace_kind, item.last_result);
  }
}

function activeChatBindings() {
  const canonical = state.chatState?.canonical_image_bindings;
  if (canonical?.bindings && Array.isArray(canonical.display_order)) {
    return canonical.display_order
      .map((label) => {
        const binding = canonical.bindings[label];
        return binding
          ? {
            ...binding,
            image_label: label,
            status: binding.active ? "active" : "removed",
          }
          : null;
      })
      .filter((item) => item?.status === "active");
  }
  return (state.chatState?.asset_bindings || []).filter((item) => item.status === "active");
}

function renderChatAssetState() {
  const target = $("context-assets");
  if (!target) return;
  const bindings = activeChatBindings();
  const bindingsByRef = new Map(bindings.map((binding) => [binding.ref, binding]));
  const focus = state.chatState?.current_focus_label || null;
  const assets = activeAssets();
  $("context-count").textContent = assets.length;
  target.innerHTML = assets.length
    ? assets.map((asset) => {
      const binding = bindingsByRef.get(asset.ref) || {
        ref: asset.ref,
        image_label: "正在加入",
        locked: false,
      };
      const locked = Boolean(binding.locked);
      return `
        <div class="chat-asset-chip ${binding.image_label === focus ? "focus" : ""} ${locked ? "locked" : ""}">
          <img src="${escapeHtml(asset.thumbnail_url || asset.image_url)}" alt="">
          <button type="button" class="chat-focus" data-focus-label="${escapeHtml(binding.image_label)}"
            ${binding.image_label === "正在加入" ? "disabled" : ""}>
            <strong>${escapeHtml(binding.image_label)} ${locked ? "🔒" : ""}</strong>
            <span>${binding.image_label === "正在加入"
              ? "等待权威会话快照"
              : locked
                ? "已参与回答 · 编号锁定"
                : (binding.image_label === focus ? "当前焦点" : "设为焦点")}</span>
          </button>
          <button type="button" class="chat-remove" data-chat-remove-ref="${escapeHtml(binding.ref)}"
            aria-label="${locked ? "图片已锁定，不能移除" : "从会话移除"}"
            title="${locked ? "该图片已参与会话回答，不能移除" : "从会话移除"}"
            ${locked ? "disabled" : ""}>${locked ? "锁定" : "×"}</button>
        </div>`;
    }).join("")
    : '<span class="empty-context">从左侧勾选图片，再点击“加入当前对话”</span>';
  if (focus) {
    target.insertAdjacentHTML(
      "beforeend",
      '<button type="button" class="quiet-button chat-clear-focus" data-clear-chat-focus>取消焦点</button>',
    );
  }
  target.querySelectorAll("[data-focus-label]").forEach((button) => {
    button.onclick = () => setChatFocus(
      button.dataset.focusLabel === focus ? null : button.dataset.focusLabel,
    );
  });
  target.querySelectorAll("[data-chat-remove-ref]").forEach((button) => {
    button.onclick = () => removeFromContext(button.dataset.chatRemoveRef);
  });
  target.querySelector("[data-clear-chat-focus]")?.addEventListener(
    "click",
    () => setChatFocus(null),
  );
}

function applyConversationState(session) {
  absorbAssets(session.active_assets || []);
  state.chatState = session.chat_state || null;
  state.chatSelectionVersion = Number(session.selection_version || 0);
  state.selectedSearchLabels = new Set(
    (state.chatState?.selected_tool_images || []).filter(
      (label) => String(label).startsWith("SEARCH_"),
    ),
  );
  const canonicalRefs = activeChatBindings()
    .map((binding) => binding.ref)
    .filter(Boolean);
  state.contextRefs = canonicalRefs.length
    ? canonicalRefs
    : (session.active_assets || [])
      .map(publicRefToLocal)
      .filter(Boolean)
      .map((item) => item.ref);
  renderContext();
}

async function syncConversationAssets({
  focusImageLabel,
  focusSpecified = false,
  operation = null,
} = {}) {
  if (!state.currentConversationId) return;
  const payload = {asset_refs: activeAssets().map(assetRequest)};
  if (focusSpecified) payload.focus_image_label = focusImageLabel;
  if (operation) {
    payload.operation_id = operation.operationId;
    payload.workspace_id = `chat:${state.currentConversationId}`;
    payload.client_sequence = operation.clientSequence;
    payload.expected_version = operation.expectedVersion;
  }
  const session = await api(`/course/conversations/${state.currentConversationId}/assets`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  applyConversationState(session);
  await loadConversations(false);
  return session;
}

async function setChatFocus(imageLabel) {
  showError("");
  try {
    await syncConversationAssets({focusImageLabel: imageLabel, focusSpecified: true});
  } catch (error) {
    showError(error.message);
  }
}

async function addToContext(ref) {
  showError("");
  if (state.contextRefs.includes(ref) || state.chatPendingRefs.has(ref)) return;
  state.chatPendingRefs.add(ref);
  renderAssetList();
  const execute = async () => {
    if (state.contextRefs.includes(ref)) return;
    if (state.contextRefs.length >= 5) {
      throw new Error("当前上下文最多 5 张图片。");
    }
    state.contextRefs.push(ref);
    renderContext();
    state.chatClientSequence += 1;
    try {
      await syncConversationAssets({
        operation: {
          operationId: operationId("chat"),
          clientSequence: state.chatClientSequence,
          expectedVersion: state.chatSelectionVersion,
        },
      });
    } catch (error) {
      if (error.status === 409 && state.currentConversationId) {
        const current = await api(`/course/conversations/${state.currentConversationId}`);
        applyConversationState(current);
      }
      throw error;
    }
  };
  const pending = state.chatQueue.then(execute, execute);
  state.chatQueue = pending.catch(() => {});
  return pending.catch((error) => showError(error.message)).finally(() => {
    state.chatPendingRefs.delete(ref);
    renderAssetList();
  });
}

async function addBatchToChat() {
  const assets = orderedBatchAssets();
  if (!assets.length) {
    return showError("请先勾选要加入当前对话的图片。");
  }
  if (!state.currentConversationId) {
    await createConversation("新对话", []);
  }
  const refs = assets.map((item) => item.ref);
  const execute = async () => {
    const merged = [...state.contextRefs];
    for (const ref of refs) {
      if (!merged.includes(ref)) merged.push(ref);
    }
    if (merged.length > 5) {
      throw new Error("当前对话最多 5 张图片，请减少批量勾选。");
    }
    state.contextRefs = merged;
    state.chatClientSequence += 1;
    try {
      return await syncConversationAssets({
        operation: {
          operationId: operationId("chat-batch"),
          clientSequence: state.chatClientSequence,
          expectedVersion: state.chatSelectionVersion,
        },
      });
    } catch (error) {
      if (error.status !== 409 || !state.currentConversationId) throw error;
      const current = await api(
        `/course/conversations/${state.currentConversationId}`,
      );
      applyConversationState(current);
      const retryMerged = [...state.contextRefs];
      for (const ref of refs) {
        if (!retryMerged.includes(ref)) retryMerged.push(ref);
      }
      if (retryMerged.length > 5) {
        throw new Error("当前对话已更新，最多仍只能保留 5 张图片。");
      }
      state.contextRefs = retryMerged;
      state.chatClientSequence += 1;
      return syncConversationAssets({
        operation: {
          operationId: operationId("chat-batch-retry"),
          clientSequence: state.chatClientSequence,
          expectedVersion: state.chatSelectionVersion,
        },
      });
    }
  };
  const pending = state.chatQueue.then(execute, execute);
  state.chatQueue = pending.catch(() => {});
  return pending
    .then((session) => {
      state.batchSelections.chat.clear();
      renderAssetList();
      showToast("已按勾选顺序加入当前对话；重复图片已跳过，批量选择已清空。");
      return session;
    })
    .catch((error) => showError(error.message));
}

$("chat-add-selected").onclick = () => addBatchToChat();

function addAssetToCurrentWorkspace(ref) {
  const asset = findAsset(ref);
  if (!asset) return;
  state.selectedRef = ref;
  const kind = currentWorkspaceKind();
  if (kind === "chat") {
    addToContext(ref);
  } else if (kind && state.workspaces[kind]) {
    enqueueWorkspaceOperation(kind, "add", {
      asset,
      localOptions: workspaceLocalOptions(kind),
    }).catch((error) => showError(error.message));
  } else {
    showToast("当前页面没有选图工作区；双击不会修改其他页面。");
  }
}

async function removeFromContext(ref) {
  const binding = activeChatBindings().find((item) => item.ref === ref);
  if (binding?.locked) {
    showToast("该图片已参与本会话，已锁定，不能移除。");
    return;
  }
  const previous = [...state.contextRefs];
  state.contextRefs = state.contextRefs.filter((item) => item !== ref);
  renderContext();
  try {
    await syncConversationAssets();
  } catch (error) {
    state.contextRefs = previous;
    renderContext();
    renderChatAssetState();
    if (String(error.message).includes("locked_image_cannot_be_removed")) {
      showToast("该图片已参与本会话，已锁定，不能移除。");
    } else {
      showError(error.message);
    }
  }
}

$("clear-context").onclick = async () => {
  const previous = [...state.contextRefs];
  const lockedRefs = new Set(
    activeChatBindings()
      .filter((item) => item.locked)
      .map((item) => item.ref),
  );
  state.contextRefs = state.contextRefs.filter((ref) => lockedRefs.has(ref));
  renderContext();
  try {
    await syncConversationAssets();
    if (lockedRefs.size) {
      showToast("已清除未锁定图片；参与过会话回答的图片会保留。");
    }
  } catch (error) {
    state.contextRefs = previous;
    renderContext();
    if (state.currentConversationId) {
      try {
        const current = await api(`/conversations/${state.currentConversationId}`);
        applyConversationState(current);
      } catch (_) {
        // Keep the restored local snapshot when the authoritative refresh also fails.
      }
    }
    showError(error.message);
  }
};

function renderFacts(data = {}) {
  const preferred = [
    "global_observation", "global_scene", "subjects", "main_subjects", "activities",
    "relations", "attributes", "visible_text_candidates", "visible_text",
    "scene_hypotheses", "uncertainties", "uncertainty", "evidence_descriptions",
    "ocr_status", "verified_text_status",
  ];
  const keys = [
    ...preferred.filter((key) => data[key] !== undefined),
    ...Object.keys(data).filter((key) => !preferred.includes(key)),
  ];
  renderSelectedTwoLayer({
    default: {
      "主题": data.global_scene || data.theme || "自定义图片",
      "简短描述": data.global_observation || data.description || "暂无结构化简述。",
      "微标签": data.subjects || data.main_subjects || [],
      "当前状态": data.status || "暂无",
    },
    details: Object.fromEntries(
      keys.map((key) => [
        LEGACY_FIELD_DISPLAY_NAMES[key] || key || "补充信息",
        data[key],
      ]),
    ),
    developer: {},
  });
}

function localizedLayerHtml(value) {
  if (value === null || value === undefined || value === "") return "暂无";
  if (Array.isArray(value)) {
    if (!value.length) return "暂无";
    return `<ul class="natural-value-list">${value.map(
      (item) => `<li>${localizedLayerHtml(item)}</li>`,
    ).join("")}</ul>`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) return "暂无";
    return `<span class="natural-object-value">${entries.map(
      ([key, item]) => `<span><b>${escapeHtml(key)}</b>：${localizedLayerHtml(item)}</span>`,
    ).join("")}</span>`;
  }
  return escapeHtml(String(value)) || "暂无";
}

function canonicalLayerRows(values = {}) {
  const entries = Object.entries(values);
  if (!entries.length) return "<dd>暂无</dd>";
  return entries
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${localizedLayerHtml(value)}</dd>`)
    .join("");
}

function renderCanonicalUserState(status, message = "") {
  const normalized = String(status || "not_started");
  const labels = {
    not_started: "尚未生成",
    generating: "正在生成",
    completed: "已生成",
    failed: "生成失败",
    waiting_for_provider: "等待模型连接",
    missing_credentials: "API Key 未配置",
    network_unavailable: "网络不可用",
    invalid_api_key: "API Key 无效",
    billing_or_quota: "额度或余额不足",
    permission: "模型权限不足",
    region_endpoint_mismatch: "地域或端点不匹配",
    rate_limit: "请求过于频繁",
    service_unavailable: "模型服务暂不可用",
    image_processing_failed: "图片处理失败",
    model_output_incomplete: "模型输出不完整",
    structured_parse_failed: "结构化解析失败",
    safety_validation_failed: "结构化安全检查未通过",
  };
  $("selected-current-status").textContent = labels[normalized] || String(status || "尚未生成");
  $("selected-canonical-message").textContent = message || (
    normalized === "completed"
      ? "Canonical 标注已保存，可展开查看结构化详情。"
      : normalized === "generating"
        ? "正在生成并校验结构化标注，请稍候。"
        : "尚未保存 Canonical 标注，可以重新尝试生成。"
  );
}

function renderSelectedTwoLayer(layer = {}, asset = null, canonicalState = null) {
  const primary = layer.default || {};
  const categories = Array.isArray(layer.categories)
    ? layer.categories
    : [];
  $("selected-category-tags").innerHTML = categories.length
    ? categories.map(
      (tag) => `<span class="canonical-category">${escapeHtml(tag)}</span>`,
    ).join("")
    : '<span class="empty-context">暂无类别</span>';
  $("selected-theme").textContent = primary["主题"] || "暂无";
  $("selected-short-description").textContent = primary["简短描述"] || "暂无";
  const tags = Array.isArray(primary["微标签"]) ? primary["微标签"] : [];
  $("selected-micro-tags").innerHTML = tags.length
    ? tags.map((tag) => `<span class="canonical-tag">${escapeHtml(tag)}</span>`).join("")
    : '<span class="empty-context">暂无</span>';
  if (canonicalState) {
    renderCanonicalUserState(canonicalState.status, canonicalState.message);
  } else if (Object.keys(primary).length) {
    const quality = primary["当前状态"] || "检查通过";
    renderCanonicalUserState("completed", `自动检查状态：${quality}。结果仍为机器暂定，可继续人工复核。`);
  } else {
    renderCanonicalUserState("not_started");
  }
  $("selected-label-detail-fields").innerHTML = canonicalLayerRows(layer.details || {});
  const developer = {
    ...(layer.developer || {}),
    ...(asset ? {
      asset_id: asset.asset_id || asset.sourceAssetId || asset.image_id,
      sha256: asset.sha256 || asset.image_sha256 || (layer.developer || {}).sha256,
    } : {}),
  };
  $("selected-developer-fields").innerHTML = canonicalLayerRows(developer);
}

function renderCanonicalPreview(items = []) {
  state.canonicalPreview = items;
  const panel = $("canonical-preview-panel");
  panel.classList.toggle("hidden", !items.length);
  $("canonical-preview-count").textContent = `${items.length} 张`;
  $("canonical-preview-grid").innerHTML = items.map((item) => {
    const layer = item.two_layer || {};
    const primary = layer.default || {};
    const tags = (primary["微标签"] || [])
      .map((tag) => `<span class="canonical-tag">${escapeHtml(tag)}</span>`)
      .join("");
    const categories = (layer.categories || [])
      .map((category) => `<span>${escapeHtml(category)}</span>`)
      .join("");
    return `
      <article class="canonical-card" data-preview-image-id="${escapeHtml(item.image_id)}">
        <img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.image_id)}">
        <div class="canonical-card-body">
          <div class="canonical-slices" aria-label="顶部类别">${categories}</div>
          <h4><small>主题</small>${escapeHtml(primary["主题"] || "暂未生成标注")}</h4>
          <p><strong>简短描述</strong>${escapeHtml(primary["简短描述"] || "该图片的自动标注尚未完成，等待重新处理或人工复核。")}</p>
          <div class="canonical-tags" aria-label="微标签">${tags || '<span class="empty-context">暂无</span>'}</div>
          <p class="canonical-state">当前状态：${escapeHtml(primary["当前状态"] || "需要复核")}</p>
          <details class="canonical-details">
            <summary>标注详情</summary>
            <dl>${canonicalLayerRows(layer.details || {})}</dl>
          </details>
          <details class="canonical-developer developer-only">
            <summary>开发者信息</summary>
            <dl>${canonicalLayerRows(layer.developer || {})}</dl>
          </details>
        </div>
      </article>`;
  }).join("");
}

async function loadCanonicalPreview() {
  try {
    const result = await api("/canonical-preview");
    renderCanonicalPreview(result.enabled ? result.items : []);
  } catch (_) {
    renderCanonicalPreview([]);
  }
}

async function selectAsset(ref) {
  const asset = findAsset(ref);
  if (!asset) return;
  showError("");
  state.selectedRef = ref;
  $("selected-image").src = asset.image_url;
  $("selected-title").textContent = asset.image_id;
  $("selected-title").title = asset.image_id;
  $("open-selected-image").href = asset.image_url;
  $("selected-media-meta").textContent = asset.width && asset.height
    ? `${asset.width} × ${asset.height} · 保持原始比例预览`
    : "保持原始比例预览";
  $("image-meta").textContent = asset.source === "system"
    ? `来源：${asset.source_split === "train" ? "训练图片库" : "验证图片库"}（系统锁定）`
    : asset.source === "library"
      ? "来源：历史示例图片"
      : asset.source === "session"
        ? "来源：当前会话临时图片"
        : "来源：自定义图片库";
  $("move-selected-asset").disabled = asset.source !== "local";
  $("delete-selected-asset").disabled = !["local", "session"].includes(asset.source);
  const activeIndex = asset.index_lifecycle?.active_provider;
  $("backfill-selected-asset").disabled = asset.source !== "local"
    || !activeIndex
    || ["indexed", "indexing", "waiting_for_provider"].includes(activeIndex.status);
  $("backfill-selected-asset").textContent = activeIndex?.status === "failed"
    ? "重试当前图片索引"
    : activeIndex?.status === "indexed"
      ? "已索引"
      : "为当前图片建立索引";
  $("analyze").disabled = !["local", "session"].includes(asset.source);
  $("save-session-asset").classList.toggle(
    "hidden",
    asset.source !== "session",
  );
  try {
    if (asset.source === "system") {
      const detail = await api(`/visual-assets/${encodeURIComponent(asset.sourceAssetId)}`);
      const layer = detail.two_layer || {};
      renderSelectedTwoLayer(layer, asset);
    } else if (asset.source === "library") {
      const detail = await api(`/library/${encodeURIComponent(asset.sourceAssetId)}`);
      renderFacts({
        ...detail.intelligence,
        ocr_status: detail.ocr.status,
        verified_text_status: detail.ocr.truth_status || "image_only_unverified_text",
      });
    } else if (asset.source === "local") {
      const detail = await api(`/local-assets/${asset.sourceAssetId}`);
      if (detail.two_layer) {
        renderSelectedTwoLayer(detail.two_layer, asset);
      } else {
        const failure = detail.analysis?.canonical_failure || {};
        const data = detail.analysis?.result?.data || {};
        if (detail.analysis_status === "failed" || failure.status === "failed") {
          renderSelectedTwoLayer({}, asset, {
            status: failure.category || "failed",
            message: failure.public_message || "本次 Canonical 标注未完成，图片和已有结果均已保留，可以重新尝试。",
          });
          $("selected-label-detail-fields").innerHTML = canonicalLayerRows({
            "处理状态": "生成失败",
            "建议": failure.retryable === false ? "请先恢复对应模型能力后再试" : "可以重新尝试生成",
          });
        } else {
          renderFacts(data.normalized_output || data.parsed_output || data);
        }
      }
    } else {
      const detail = await api(
        `/session-assets/${encodeURIComponent(asset.sourceAssetId)}?conversation_id=${encodeURIComponent(asset.conversationId || state.currentConversationId)}`,
      );
      if (detail.two_layer) {
        renderSelectedTwoLayer(detail.two_layer, asset);
      } else if (detail.canonical_status === "failed" || detail.canonical_failure) {
        const failure = detail.canonical_failure || {};
        renderSelectedTwoLayer({}, asset, {
          status: failure.category || "failed",
          message: failure.public_message || "本次 Canonical 标注未完成，可以重新尝试。",
        });
      } else {
        renderFacts({
          status: "仅在当前会话使用，尚未生成 Canonical 标注。",
        });
      }
    }
    if (asset.source !== "system") {
      const developer = {
        asset_id: asset.asset_id || asset.sourceAssetId || asset.image_id,
        sha256: asset.sha256 || asset.image_sha256 || "暂无",
      };
      $("selected-developer-fields").innerHTML = canonicalLayerRows(developer);
    }
    syncAssetListState();
  } catch (error) {
    showError(error.message);
  }
}

$("add-context").onclick = () => {
  if (!state.selectedRef) return showError("请先从左侧选择一张图片。");
  addAssetToCurrentWorkspace(state.selectedRef);
};

$("move-selected-asset").onclick = async () => {
  const asset = findAsset(state.selectedRef);
  if (!asset || asset.source !== "local") return showToast("系统库资产不能移动。");
  const candidates = state.libraries.filter(
    (item) => item.library_type === "user_custom"
      && !item.locked
      && item.library_id !== asset.library_id,
  );
  if (!candidates.length) return showToast("没有其他可移动到的自定义图片库。");
  const names = candidates.map((item) => item.display_name || item.name).join("、");
  const chosen = prompt(`输入目标自定义图片库名称：${names}`);
  const target = candidates.find((item) => (item.display_name || item.name) === chosen);
  if (!target) return;
  try {
    await api(`/local-assets/${encodeURIComponent(asset.sourceAssetId)}/move`, {
      method: "POST",
      body: JSON.stringify({target_library_id: target.library_id}),
    });
    state.selectedRef = null;
    await loadLibraries();
    await loadAssets();
  } catch (error) {
    showError(error.message);
  }
};

$("delete-selected-asset").onclick = async () => {
  const asset = findAsset(state.selectedRef);
  if (!asset || !["local", "session"].includes(asset.source)) {
    return showToast("系统库资产不能删除。");
  }
  if (!confirm(`移除图片“${asset.image_id}”？`)) return;
  try {
    if (asset.source === "session") {
      await api(
        `/session-assets/${encodeURIComponent(asset.sourceAssetId)}?conversation_id=${encodeURIComponent(asset.conversationId || state.currentConversationId)}`,
        {method: "DELETE"},
      );
      state.sessionAssets = state.sessionAssets.filter((item) => item.ref !== asset.ref);
    } else {
      await api(`/local-assets/${encodeURIComponent(asset.sourceAssetId)}`, {method: "DELETE"});
    }
    state.selectedRef = null;
    state.assets = state.assets.filter((item) => item.ref !== asset.ref);
    await loadLibraries();
    await loadAssets();
  } catch (error) {
    showError(error.message);
  }
};

let assetFilterTimer = null;
function scheduleAssetFilter() {
  window.clearTimeout(assetFilterTimer);
  assetFilterTimer = window.setTimeout(() => {
    state.libraryPage = 1;
    loadAssets();
  }, 220);
}
$("asset-filter").oninput = scheduleAssetFilter;

async function loadLibraries() {
  const current = $("library-select").value;
  const result = await api("/visual-libraries");
  state.libraries = result.items || [];
  $("library-select").innerHTML = state.libraries.map((item) => {
    const name = item.display_name || item.name || item.library_id;
    const suffix = item.locked ? ` 🔒 ${item.asset_count ?? 0}` : ` · ${item.asset_count ?? 0}`;
    return `<option value="${escapeHtml(item.library_id)}">${escapeHtml(name + suffix)}</option>`;
  }).join("");
  $("library-select").value = state.libraries.some((item) => item.library_id === current)
    ? current
    : (state.libraries[0]?.library_id || "default");
  updateLibraryControls();
}

function currentLibrary() {
  return state.libraries.find((item) => item.library_id === $("library-select").value) || null;
}

function updateLibraryControls() {
  const locked = Boolean(currentLibrary()?.locked);
  $("library-lock-notice").classList.toggle("hidden", !locked);
  $("rename-library").disabled = locked;
  $("delete-library").disabled = locked;
  const persistentImportBlocked = (
    locked && $("import-mode").value === "custom_library"
  );
  $("file-input").disabled = persistentImportBlocked;
  $("upload-button").classList.toggle("disabled", persistentImportBlocked);
  $("upload-button").setAttribute(
    "aria-disabled",
    persistentImportBlocked ? "true" : "false",
  );
  $("import-mode-hint").textContent = $("import-mode").value === "session_only"
    ? "仅存放在当前会话临时区，不写入图库或持久 E1；页面刷新、会话删除或过期后按会话生命周期清理。"
    : "持久保存到项目托管存储；刷新页面或重启服务后仍可使用，并进入当前自定义图库与 E1。";
}

async function loadAssets({providerProjectionSequence = null} = {}) {
  const libraryId = $("library-select").value || "system_train";
  state.libraryLoadSequence += 1;
  const loadSequence = state.libraryLoadSequence;
  const query = new URLSearchParams({
    page: String(state.libraryPage),
    page_size: "40",
    q: $("asset-filter").value.trim(),
    label_status: $("label-filter").value,
    sort: $("asset-sort").value,
  });
  const result = await api(`/visual-libraries/${encodeURIComponent(libraryId)}/assets?${query}`);
  if (
    loadSequence !== state.libraryLoadSequence
    || libraryId !== $("library-select").value
    || (
      providerProjectionSequence !== null
      && providerProjectionSequence !== state.providerProjectionSequence
    )
  ) return;
  const library = currentLibrary();
  state.libraryPage = result.page || 1;
  state.libraryPageCount = Math.max(1, result.page_count || 1);
  state.libraryTotal = result.total || 0;
  if (library?.library_type === "system_locked") {
    state.libraryAssets = result.items.map(normalizedSystemAsset);
    state.localAssets = [];
  } else {
    state.libraryAssets = [];
    state.localAssets = result.items.map(normalizedLocalAsset);
  }
  state.assets = [
    ...new Map(
      [...state.assets, ...state.libraryAssets, ...state.localAssets, ...state.sessionAssets]
        .map((item) => [item.ref, item]),
    ).values(),
  ];
  $("library-page").textContent = `第 ${state.libraryPage} / ${state.libraryPageCount} 页`;
  $("library-prev").disabled = state.libraryPage <= 1;
  $("library-next").disabled = state.libraryPage >= state.libraryPageCount;
  renderAssetList();
  renderContext();
  if (!state.selectedRef && state.assets.length) await selectAsset(state.assets[0].ref);
}

$("library-select").onchange = () => {
  state.libraryPage = 1;
  updateLibraryControls();
  loadAssets();
};
$("label-filter").onchange = () => { state.libraryPage = 1; loadAssets(); };
$("asset-sort").onchange = () => { state.libraryPage = 1; loadAssets(); };
$("import-mode").onchange = updateLibraryControls;
$("library-prev").onclick = () => {
  if (state.libraryPage > 1) {
    state.libraryPage -= 1;
    loadAssets();
  }
};
$("library-next").onclick = () => {
  if (state.libraryPage < state.libraryPageCount) {
    state.libraryPage += 1;
    loadAssets();
  }
};
$("new-library").onclick = async () => {
  const name = prompt("图片库名称");
  if (!name) return;
  try {
    await api("/libraries", {method: "POST", body: JSON.stringify({name})});
    await loadLibraries();
    const created = state.libraries.find((item) => (item.display_name || item.name) === name);
    if (created) $("library-select").value = created.library_id;
    state.libraryPage = 1;
    await loadAssets();
  } catch (error) { showError(error.message); }
};

$("rename-library").onclick = async () => {
  const library = currentLibrary();
  if (!library || library.locked) return showToast("系统锁定库不能重命名。");
  const name = prompt("新的图片库名称", library.display_name || library.name || "");
  if (!name) return;
  try {
    await api(`/libraries/${encodeURIComponent(library.library_id)}`, {
      method: "PATCH",
      body: JSON.stringify({name}),
    });
    await loadLibraries();
    $("library-select").value = library.library_id;
    updateLibraryControls();
  } catch (error) {
    showError(error.message);
  }
};

$("delete-library").onclick = async () => {
  const library = currentLibrary();
  if (!library || library.locked) return showToast("系统锁定库不能删除。");
  if (!confirm(`删除自定义图片库“${library.display_name || library.name}”及其本地资产？`)) return;
  try {
    await api(`/libraries/${encodeURIComponent(library.library_id)}`, {method: "DELETE"});
    await loadLibraries();
    state.libraryPage = 1;
    await loadAssets();
  } catch (error) {
    showError(error.message);
  }
};

$("file-input").onchange = async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  showError("");
  try {
    const encoded = await Promise.all(files.map((file) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve({name: file.name, content_base64: String(reader.result).split(",")[1]});
      reader.onerror = reject;
      reader.readAsDataURL(file);
    })));
    if ($("import-mode").value === "session_only") {
      if (!state.currentConversationId) {
        throw new Error("请先创建或选择一个会话，再导入仅当前会话使用的图片。");
      }
      const result = await api("/session-assets/import", {
        method: "POST",
        body: JSON.stringify({
          conversation_id: state.currentConversationId,
          files: encoded,
        }),
      });
      state.sessionAssets = [
        ...new Map(
          [
            ...state.sessionAssets,
            ...result.items.map(normalizedSessionAsset),
          ].map((item) => [item.ref, item]),
        ).values(),
      ];
      state.assets = [
        ...new Map(
          [...state.assets, ...state.sessionAssets].map((item) => [item.ref, item]),
        ).values(),
      ];
      renderAssetList();
      showToast(`已加入当前会话：${result.count} 张；不会写入持久图片库。`);
    } else {
      let target = currentLibrary();
      if (!target || target.library_type !== "user_custom" || target.locked) {
        target = state.libraries.find((item) => item.library_type === "user_custom" && !item.locked);
      }
      if (!target) throw new Error("请先创建一个自定义图片库。");
      const result = await api("/imports", {
        method: "POST",
        body: JSON.stringify({library_id: target.library_id, files: encoded}),
      });
      $("library-select").value = target.library_id;
      await loadLibraries();
      $("library-select").value = target.library_id;
      await loadAssets();
      const cloudUpdate = result.cloud_index_update || {};
      if (cloudUpdate.status === "completed") {
        showToast(`导入完成：${result.assets.length} 张；${cloudUpdate.message}。`);
        await loadCloudIndexStatus();
      } else if (cloudUpdate.status === "failed") {
        showToast(cloudUpdate.message || "图片已保留，但云端索引更新失败。", 6500);
      } else {
        showToast(`导入完成：${result.assets.length} 张；重复内容按 SHA-256 去重。`);
      }
    }
  } catch (error) {
    showError(error.message);
  } finally {
    event.target.value = "";
  }
};

$("save-session-asset").onclick = async () => {
  const asset = findAsset(state.selectedRef);
  if (!asset || asset.source !== "session") return;
  let target = currentLibrary();
  if (!target || target.library_type !== "user_custom" || target.locked) {
    target = state.libraries.find(
      (item) => item.library_type === "user_custom" && !item.locked,
    );
  }
  if (!target) return showError("请先创建一个自定义图片库。");
  setBusy($("save-session-asset"), true, "保存中");
  try {
    const result = await api(
      `/session-assets/${encodeURIComponent(asset.sourceAssetId)}/persist`,
      {
        method: "POST",
        body: JSON.stringify({
          conversation_id: asset.conversationId || state.currentConversationId,
          library_id: target.library_id,
        }),
      },
    );
    $("library-select").value = target.library_id;
    await loadLibraries();
    $("library-select").value = target.library_id;
    await loadAssets();
    showToast(
      result.persistent_asset?.duplicate
        ? "该图片已在自定义图库中，已复用原资产。"
        : "已保存到自定义图片库；临时副本仍按会话生命周期保留。",
    );
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy($("save-session-asset"), false);
  }
};

function publicRefToLocal(sessionAsset) {
  const ref = sessionAsset.ref || `${sessionAsset.source || "library"}:${sessionAsset.source_asset_id || sessionAsset.image_id || sessionAsset.asset_id}`;
  return findAsset(ref);
}

function messageDisplay(content) {
  return publicDisplayText(content, "本次回答未成功，请稍后重试。");
}

function messageTechnicalDetails(message) {
  if (message.role !== "assistant" || !message.content || typeof message.content !== "object") return "";
  const content = message.content;
  let status = content.repair_level || "";
  if (content.status === "clarification_required") status = "clarification";
  else if (!status && content.model_called === false) status = "deterministic";
  const refs = content.answer?.image_references || [];
  const labels = refs.map((item) => typeof item === "string" ? item : item.image_label).filter(Boolean);
  const errors = (content.contract_errors || content.model_contract_errors || []).join("; ");
  return [
    status ? `处理层级：${status}` : "",
    labels.length ? `图片引用：${labels.join(" / ")}` : "",
    errors ? `技术错误：${errors}` : "",
    message.trace_id ? `Trace：${message.trace_id}` : "",
  ].filter(Boolean).join("\n");
}

function messageImageCards(content) {
  if (!content || typeof content !== "object") return [];
  if (Array.isArray(content.image_cards)) return content.image_cards;
  if (Array.isArray(content.search_results)) return content.search_results;
  if (Array.isArray(content.results) && content.results.some((item) => item.tool_image_label)) {
    return content.results;
  }
  return [];
}

function activeSearchCard(item) {
  return (state.chatState?.last_search_results || []).some(
    (active) => active.tool_image_label === item.tool_image_label
      && (active.asset_id || active.image_url) === (item.asset_id || item.image_url),
  );
}

function renderMessageImageCards(content) {
  const cards = messageImageCards(content);
  if (!cards.length) return "";
  return `
    <div class="chat-search-grid">
      ${cards.map((item) => {
        const label = item.tool_image_label || `SEARCH_${item.rank || 1}`;
        const active = activeSearchCard(item);
        const selected = state.selectedSearchLabels.has(label);
        return `
          <article class="chat-search-card ${selected ? "selected" : ""} ${active ? "" : "expired"}">
            <a href="${escapeHtml(item.content_url || item.image_url || "#")}" target="_blank" rel="noopener" aria-label="查看 ${escapeHtml(label)}">
              ${resultMediaMarkup(item, label)}
            </a>
            <div class="chat-search-card-body">
              <strong>${escapeHtml(label)}</strong>
              <small>${escapeHtml(item.display_name || item.asset_id || "")}</small>
              <span>${active ? "临时搜索结果" : "已被后续搜索替换"}</span>
              <div class="chat-search-actions">
                <button type="button" data-search-select="${escapeHtml(label)}" ${active ? "" : "disabled"}>${selected ? "取消选中" : "选中"}</button>
                <button type="button" data-search-import="chat_context" data-search-label="${escapeHtml(label)}" ${active ? "" : "disabled"}>加入会话</button>
                <button type="button" data-search-import="generation_workspace" data-search-label="${escapeHtml(label)}" ${active ? "" : "disabled"}>用于生成</button>
                <button type="button" data-search-import="compare_workspace" data-search-label="${escapeHtml(label)}" ${active ? "" : "disabled"}>用于选图排序</button>
              </div>
            </div>
          </article>`;
      }).join("")}
    </div>`;
}

async function persistSearchSelection() {
  if (!state.currentConversationId) return;
  await api(
    `/course/conversations/${state.currentConversationId}/search-results/selection`,
    {
      method: "PUT",
      body: JSON.stringify({
        search_labels: [...state.selectedSearchLabels],
      }),
    },
  );
}

async function importSearchCards(destination, clickedLabel) {
  if (!state.currentConversationId) return;
  const selected = [...state.selectedSearchLabels];
  const labels = selected.includes(clickedLabel) && selected.length
    ? selected
    : [clickedLabel];
  const result = await api(
    `/course/conversations/${state.currentConversationId}/search-results/import`,
    {
      method: "POST",
      body: JSON.stringify({
        search_labels: labels,
        destination,
      }),
    },
  );
  $("vqa-result").textContent = result.display_text || "已完成。";
  if (destination === "chat_context") {
    applyConversationState(result.session);
    renderMessages(result.session.messages || []);
    await loadConversations(false);
    return;
  }
  await loadFunctionWorkspaces();
  navigate(destination === "generation_workspace" ? "generate" : "rank");
}

function renderMessages(messages = []) {
  const messagesNode = $("chat-messages");
  const distanceFromBottom =
    messagesNode.scrollHeight - messagesNode.scrollTop - messagesNode.clientHeight;
  const shouldStickToBottom = messagesNode.childElementCount === 0 || distanceFromBottom < 72;
  $("chat-welcome").classList.toggle("hidden", messages.length > 0);
  $("chat-messages").innerHTML = messages.map((message) => {
    const technical = DEBUG_UI ? messageTechnicalDetails(message) : "";
    return `
      <article class="message ${message.role === "user" ? "user" : "assistant"}">
        <div class="message-public-text">${escapeHtml(messageDisplay(message.content)).replaceAll("\n", "<br>")}</div>
        ${message.role === "assistant" ? renderMessageImageCards(message.content) : ""}
        ${technical ? `<details class="message-technical"><summary>技术详情</summary><pre>${escapeHtml(technical)}</pre></details>` : ""}
      </article>`;
  }).join("");
  document.querySelectorAll("[data-search-select]").forEach((button) => {
    button.onclick = async () => {
      const label = button.dataset.searchSelect;
      if (state.selectedSearchLabels.has(label)) {
        state.selectedSearchLabels.delete(label);
      } else {
        state.selectedSearchLabels.add(label);
      }
      try {
        await persistSearchSelection();
        const session = await api(`/conversations/${state.currentConversationId}`);
        applyConversationState(session);
        renderMessages(session.messages || []);
      } catch (error) {
        showError(error.message);
      }
    };
  });
  document.querySelectorAll("[data-search-import]").forEach((button) => {
    button.onclick = async () => {
      setBusy(button, true, "处理中");
      try {
        await importSearchCards(
          button.dataset.searchImport,
          button.dataset.searchLabel,
        );
      } catch (error) {
        showError(error.message);
      } finally {
        setBusy(button, false);
      }
    };
  });
  if (shouldStickToBottom) {
    requestAnimationFrame(() => {
      messagesNode.scrollTop = messagesNode.scrollHeight;
    });
  }
}

function renderConversations() {
  $("conversation-list").innerHTML = state.conversations.length
    ? state.conversations.map((item) => `
      <div class="conversation-row ${item.conversation_id === state.currentConversationId ? "active" : ""}">
        <button class="conversation-item ${item.conversation_id === state.currentConversationId ? "active" : ""}" data-conversation="${escapeHtml(item.conversation_id)}">
          <strong>${escapeHtml(item.title || "未命名对话")}</strong>
          <small>${(item.active_assets || []).length} 图 · ${(item.messages || []).length} 条消息</small>
        </button>
        <button class="conversation-delete" type="button" data-delete-conversation="${escapeHtml(item.conversation_id)}" aria-label="删除会话：${escapeHtml(item.title || "未命名对话")}" title="删除会话">删除</button>
      </div>`).join("")
    : '<span class="empty-context">暂无对话</span>';
  document.querySelectorAll("[data-conversation]").forEach((button) => {
    button.onclick = () => switchConversation(button.dataset.conversation);
  });
  document.querySelectorAll("[data-delete-conversation]").forEach((button) => {
    button.onclick = async () => {
      try {
        await deleteConversation(button.dataset.deleteConversation);
      } catch (error) {
        showError(error.message);
      }
    };
  });
  requestAnimationFrame(() => {
    $("conversation-list")
      .querySelector(".conversation-row.active")
      ?.scrollIntoView({block: "nearest"});
  });
}

async function loadConversations(renderCurrent = true) {
  const result = await api("/conversations");
  state.conversations = result.items;
  renderConversations();
  if (renderCurrent && state.currentConversationId) {
    const exists = state.conversations.some((item) => item.conversation_id === state.currentConversationId);
    if (exists) await switchConversation(state.currentConversationId);
  }
}

async function switchConversation(id) {
  try {
    if (state.currentConversationId !== id) {
      state.sessionAssets = [];
      state.assets = state.assets.filter((item) => item.source !== "session");
    }
    const session = await api(`/conversations/${id}`);
    state.currentConversationId = id;
    applyConversationState(session);
    renderMessages(session.messages || []);
    $("conversation-meta").textContent = `${session.title || "对话"} · ${id}`;
    renderConversations();
  } catch (error) {
    showError(error.message);
  }
}

async function createConversation(title = "新对话", assetRefs = null) {
  const session = await api("/course/conversations", {
    method: "POST",
    body: JSON.stringify({
      title,
      asset_refs: assetRefs === null ? activeAssets().map(assetRequest) : assetRefs,
    }),
  });
  state.currentConversationId = session.conversation_id;
  applyConversationState(session);
  renderMessages([]);
  $("conversation-meta").textContent = `${session.title} · ${session.conversation_id}`;
  await loadConversations(false);
  return session;
}

async function deleteConversation(conversationId) {
  const target = state.conversations.find(
    (item) => item.conversation_id === conversationId,
  );
  const confirmed = window.confirm(
    `确定删除“${target?.title || "这个会话"}”吗？聊天记录和本会话中的图片绑定将被删除，图库中的原始图片不会受到影响。`,
  );
  if (!confirmed) return false;
  const deletingCurrent = state.currentConversationId === conversationId;
  await api(`/conversations/${conversationId}`, {method: "DELETE"});
  await loadConversations(false);
  if (!deletingCurrent) {
    renderConversations();
    return true;
  }
  const next = state.conversations[0];
  if (next) {
    await switchConversation(next.conversation_id);
  } else {
    await createConversation("新对话", []);
  }
  return true;
}

$("new-conversation").onclick = async () => {
  try { await createConversation("新对话", []); } catch (error) { showError(error.message); }
};

$("content-mode").addEventListener("change", () => {
  state.contentTypeUserSelected = true;
  state.contentLengthUserEdited = false;
  applyContentLengthProfile({resetValue: true});
});
$("content-length").addEventListener("input", () => {
  state.contentLengthUserEdited = true;
});
$("content-length").addEventListener("change", () => {
  normalizeContentLengthInput({notify: true});
});
$("content-length").addEventListener("blur", () => {
  normalizeContentLengthInput({notify: true});
});
$("search-query").addEventListener("input", updateDetectedSearchMode);
$("compare-action").addEventListener("change", updateCompareControls);
updateCompareControls();
updateDetectedSearchMode();

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.onclick = () => {
    $("vqa-question").value = button.dataset.prompt;
    $("vqa-question").focus();
  };
});

$("vqa-form").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.submitter;
  const message = $("vqa-question").value.trim();
  if (!message) return;
  showError("");
  setBusy(button, true, "发送中");
  try {
    if (!state.currentConversationId) await createConversation(message.slice(0, 36));
    const request = {
      method: "POST",
      body: JSON.stringify({
        conversation_id: state.currentConversationId,
        message,
      }),
    };
    const result = isExactRuntimeModelIdentityQuestion(message)
      ? await api("/course/chat", {...request, timeoutMs: 30000})
      : await modelApi(
        "/course/chat",
        request,
        {button, activeLabel: "发送中"},
      );
    state.currentConversationId = result.conversation_id;
    state.chatState = result.chat_state || state.chatState;
    state.selectedSearchLabels = new Set(
      (state.chatState?.selected_tool_images || []).filter(
        (label) => String(label).startsWith("SEARCH_"),
      ),
    );
    state.contextRefs = (result.active_assets || [])
      .map(publicRefToLocal)
      .filter(Boolean)
      .map((item) => item.ref);
    renderContext();
    renderMessages(result.messages);
    $("vqa-result").textContent = publicDisplayText(
      result.response,
      "本次回答未成功，请稍后重试。",
    );
    $("vqa-question").value = "";
    recordTrace(result);
    await loadConversations(false);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(button, false);
  }
};

function renderStructuredGeneration(payload) {
  const value = payload.content || {};
  const finalText = publicDisplayText(
    payload,
    "本次内容生成未成功，请稍后重试。",
  );
  const length = payload.length_contract || {};
  const contractPassed = Boolean(payload.product_contract_valid);
  const fallbackLabels = {
    risk_generalization: "已做文字风险泛化",
    unverified_text_visual_generalization: "已做未核验文字泛化",
    safe_short_multiview_bridge: "已做安全短文收束",
    model_authored_bounded_completion: "已做模型续写补全",
    target_length_safe_compression: "已按目标长度安全收束",
    friendly_failure: "生成未通过",
  };
  const fallbackLabel = payload.fallback_applied
    ? fallbackLabels[payload.fallback_source] || (contractPassed ? "已做安全修复" : "生成未通过")
    : "模型正文";
  const technical = {
    prompt: payload.prompt_candidate,
    model: payload.result ? {
      name: payload.result.model,
      revision: payload.result.model_revision,
    } : null,
    contract_valid: contractPassed,
    length_contract: length,
    image_coverage: payload.image_coverage,
    repair_applied: payload.repair_applied,
    risk_sanitized: payload.risk_sanitized,
    fallback_source: payload.fallback_source,
    contract_errors: payload.contract_errors || [],
    intent_resolution: payload.resolved_options?.intent_resolution || null,
    story_structure: payload.story_structure || value.story_structure || null,
    evidence: value.evidence || [],
    uncertainty: value.uncertainty || [],
    request_id: payload.request_id,
  };
  $("content-result").classList.remove("empty-result");
  $("content-result").classList.toggle("generation-failed", !contractPassed);
  $("content-result").innerHTML = `
    <p id="content-final-text" class="generated-final-text">${escapeHtml(finalText)}</p>
    <div class="generation-meta">
      <span>${escapeHtml(payload.resolved_options?.content_type || payload.options?.content_type || "content")}</span>
      <span>${contractPassed ? `${length.actual} 字 · 目标 ${length.target}` : "未形成可交付正文"}</span>
      <span>${escapeHtml(fallbackLabel)}</span>
    </div>
    ${DEBUG_UI ? `
      <details class="generation-technical">
        <summary>技术详情与 Trace</summary>
        <pre>${escapeHtml(pretty(technical))}</pre>
      </details>` : ""}
  `;
}

$("content-form").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.submitter;
  const assets = workspaceAssets("generation");
  if (!assets.length) return showError("请先在生成工作区加入 1-5 张图片。");
  setBusy(button, true, "生成中");
  showError("");
  try {
    const targetLength = state.contentLengthUserEdited
      ? normalizeContentLengthInput({notify: true})
      : null;
    const result = await modelApi("/course/generate", {
      method: "POST",
      body: JSON.stringify({
        asset_refs: assets.map(assetRequest),
        content_type: $("content-mode").value,
        natural_language_request: $("content-request").value.trim(),
        content_type_source: state.contentTypeUserSelected
          ? "explicit_user_selection"
          : "default_value",
        content_type_user_selected: state.contentTypeUserSelected,
        target_length: targetLength,
        organization: $("content-organization").value,
        importance: assets.map((item) => item.ref),
        call_source: "standalone_workspace",
        workspace_id: state.workspaces.generation.workspace_id,
      }),
    }, {button, activeLabel: "生成中"});
    renderStructuredGeneration(result);
    if (result.public_hint) showToast(result.public_hint);
    recordTrace(result);
    await persistWorkspace(
      "generation",
      workspaceResultSnapshot("generation", result),
    );
  } catch (error) { showError(error.message); }
  finally { setBusy(button, false); }
};

function retrievalDisplayFilename(item) {
  const authority = item.display_name
    || item.original_filename
    || item.filename
    || item.asset_id
    || item.image_id
    || "unknown_asset";
  const safe = String(authority).trim().replaceAll("\\", "/").split("/").pop();
  return safe || String(item.asset_id || item.image_id || "unknown_asset").slice(0, 48);
}

function renderSearchResults(payload) {
  const modeLabels = {text: "文字检索", image: "以图搜图", hybrid: "图文联合检索"};
  const fallback = payload.fallback_used ? " · 已使用备用检索" : "";
  $("search-summary").textContent = `${modeLabels[payload.mode] || "智能检索"} · ${payload.results.length} 条${fallback}`;
  $("search-results").innerHTML = payload.results.map((item) => {
    const score = Number(item.score || 0);
    const percent = Math.max(0, Math.min(100, score * 100));
    const filename = retrievalDisplayFilename(item);
    return `
      <article class="result-card">
        ${resultMediaMarkup(item, `检索结果 ${item.rank}`)}
        <div class="result-card-body">
          <span class="result-rank">结果 ${item.rank}</span>
          <strong class="result-filename" title="${escapeHtml(filename)}">${escapeHtml(filename)}</strong>
          <small class="result-source">${escapeHtml(item.library_name || item.library_id || "未知图片库")} · ${escapeHtml(item.source_type || item.source || "persistent")}</small>
          <div class="score-row"><span>匹配度</span><b>${score.toFixed(4)}</b></div>
          <div class="score-bar"><span style="width:${percent}%"></span></div>
          <p class="result-reason">${escapeHtml(item.reason || "-")}</p>
        </div>
      </article>`;
  }).join("");
}

$("search-form").onsubmit = async (event) => {
  event.preventDefault();
  const button = event.submitter;
  const query = $("search-query").value.trim();
  const assets = workspaceAssets("retrieval");
  const mode = detectedSearchMode();
  if (!mode) return showError("请输入检索文字，或在检索工作区加入查询图片。");
  setBusy(button, true, "检索中");
  showError("");
  try {
    const body = {};
    if (mode !== "image") body.query_text = query;
    if (mode !== "text") body.query_asset_refs = assets.map(assetRequest);
    const scopeValue = $("search-library-scope").value;
    body.library_scope = scopeValue;
    if (scopeValue === "current_library") {
      body.current_library_id = currentLibrary()?.library_id || null;
    }
    body.top_k = Number($("search-top-k").value || 5);
    body.exclude_query_images = $("search-exclude-query").checked;
    body.call_source = "standalone_workspace";
    body.workspace_id = state.workspaces.retrieval.workspace_id;
    const result = await modelApi(
      "/course/retrieve",
      {
        method: "POST",
        body: JSON.stringify(body),
        timeoutMs: 90000,
      },
      {
        capability: "retrieval",
        button,
        activeLabel: "检索中",
      },
    );
    renderSearchResults(result);
    recordTrace(result);
    await loadStatus({renderSystemCards: false});
    await persistWorkspace(
      "retrieval",
      workspaceResultSnapshot("retrieval", result),
    );
  } catch (error) { showError(error.message); }
  finally { setBusy(button, false); }
};

function renderRanking(payload) {
  const fullRanking = Array.isArray(payload.ranking) ? payload.ranking : [];
  const selectedRanking = Array.isArray(payload.selected)
    ? payload.selected.map((item, index) => ({
      ...item,
      rank: Number(item.rank || index + 1),
    }))
    : [];
  const visibleRanking = payload.action === "select"
    ? (fullRanking.length ? fullRanking : selectedRanking)
      .slice(0, Math.max(1, Number(payload.select_count || 1)))
    : fullRanking;
  $("compare-result").classList.remove("empty-result");
  $("compare-result").innerHTML = `
    <section class="ranking-conclusion">
      <h3>排序结论</h3>
      <div>${escapeHtml(publicDisplayText(payload, "本次未能给出可靠排序。"))}</div>
    </section>
    <div class="ranking-list">
      ${visibleRanking.map((item) => `
        <div class="ranking-row">
          <span class="ranking-position">${item.rank}</span>
          <span class="ranking-copy"><strong>${escapeHtml(item.image_label || `第 ${item.rank} 张`)}</strong><small>${escapeHtml(item.reason || "-")}</small></span>
        </div>`).join("")}
    </div>
    ${DEBUG_UI ? `<details class="result-section"><summary>技术详情</summary><pre>${escapeHtml(pretty({
      repair_level: payload.repair_level,
      errors: payload.model_contract_errors,
      trace_id: payload.request_id,
      prompt: payload.prompt_candidate,
      full_ranking: fullRanking,
    }))}</pre></details>` : ""}`;
}

$("compare-form").onsubmit = async (event) => {
  event.preventDefault();
  const assets = workspaceAssets("compare");
  const criterion = $("compare-instruction").value.trim();
  if (assets.length < 2) return showError("请在比较工作区加入至少 2 张图片。");
  if (!criterion) return showError("请输入比较标准。");
  const button = event.submitter;
  setBusy(button, true, "处理中");
  try {
    const result = await modelApi("/course/compare", {
      method: "POST",
      body: JSON.stringify({
        criterion,
        scenario: criterion,
        action: $("compare-action").value,
        select_count: $("compare-action").value === "select"
          ? Number($("compare-select-count").value || 1)
          : 1,
        asset_refs: assets.map(assetRequest),
        call_source: "standalone_workspace",
        workspace_id: state.workspaces.compare.workspace_id,
      }),
    }, {button, activeLabel: "处理中"});
    renderRanking(result);
    recordTrace(result);
    await persistWorkspace(
      "compare",
      workspaceResultSnapshot("compare", result),
    );
  } catch (error) { showError(error.message); }
  finally { setBusy(button, false); }
};

$("analyze").onclick = async () => {
  const asset = findAsset(state.selectedRef);
  if (!asset) return showError("请先选择图片。");
  if (!["local", "session"].includes(asset.source)) {
    return showToast("系统 Train/Val 标签为只读；当前页面只展示既有 Canonical。");
  }
  renderCanonicalUserState("generating");
  setBusy($("analyze"), true, "标注中");
  try {
    const endpoint = asset.source === "session"
      ? `/session-assets/${encodeURIComponent(asset.sourceAssetId)}/canonical-label?conversation_id=${encodeURIComponent(asset.conversationId || state.currentConversationId)}`
      : `/local-assets/${encodeURIComponent(asset.sourceAssetId)}/canonical-label`;
    const result = await modelApi(
      endpoint,
      {method: "POST"},
      {button: $("analyze"), activeLabel: "标注中"},
    );
    renderSelectedTwoLayer(result.two_layer || {}, asset);
    recordTrace(result);
    await loadAssets();
    showToast(
      result.recovery_mode === "minimal_safe_contract"
        ? "Canonical 完整结构连续截断后，已由冻结的最小安全合同完成恢复。"
        : result.recovery_mode === "expanded_full_contract"
          ? "Canonical 标注已完成；检测到明确截断后使用扩展输出预算恢复。"
          : "Canonical 标注已完成并通过结构、文字安全与公开输出检查。",
    );
  } catch (error) {
    showError(error.message);
    if (asset.source === "local") {
      try {
        await selectAsset(state.selectedRef);
      } catch (_) {
        renderCanonicalUserState("failed", error.message);
      }
    } else {
      renderCanonicalUserState("failed", error.message);
    }
  }
  finally { setBusy($("analyze"), false); }
};

$("model-access-entry").onclick = () => openModelAccess();
$("model-access-close").onclick = closeModelAccess;
$("provider-continue-browsing").onclick = closeModelAccess;
document.querySelectorAll('input[name="provider-mode"]').forEach((input) => {
  input.addEventListener("change", () => {
    updateProviderSections(input.value);
  });
});
document.querySelectorAll('input[name="credential-source"]').forEach((input) => {
  input.addEventListener("change", updateCredentialPanels);
});
$("provider-api-key-toggle").onclick = () => {
  const input = $("provider-api-key");
  const reveal = input.type === "password";
  input.type = reveal ? "text" : "password";
  $("provider-api-key-toggle").textContent = reveal ? "隐藏" : "显示";
  $("provider-api-key-toggle").setAttribute("aria-pressed", String(reveal));
  input.focus();
};
$("provider-api-key-clear").onclick = () => {
  $("provider-api-key").value = "";
  $("provider-api-key").type = "password";
  $("provider-api-key-toggle").textContent = "显示";
  $("provider-api-key-toggle").setAttribute("aria-pressed", "false");
  $("provider-api-key-summary").textContent = "输入框已清空；后端已保存凭据不会因此自动删除。";
  $("provider-api-key").focus();
};
function updateEndpointFields() {
  const mode = $("provider-endpoint-mode").value;
  $("provider-workspace-row").classList.toggle("hidden", mode !== "workspace");
  $("provider-api-host-row").classList.toggle("hidden", mode !== "custom");
}
$("provider-endpoint-mode").addEventListener("change", updateEndpointFields);
updateEndpointFields();

async function revalidateProvider({source = "manual"} = {}) {
  if (state.providerRevalidateInFlight) return state.providerRevalidateInFlight;
  const switchSequence = state.providerSwitchSequence;
  const button = $("provider-revalidate");
  setBusy(button, true, "重新连接中");
  state.providerRevalidateInFlight = (async () => {
    try {
      const result = await api("/providers/access/revalidate", {
        method: "POST",
        body: "{}",
      });
      if (switchSequence !== state.providerSwitchSequence) return null;
      renderProviderAccess(result.provider);
      await refreshProviderAwareProjection({renderSystemCards: false});
      const ready = (result.provider.connection_state || result.provider.state) === "READY";
      showToast(
        ready
          ? "连接状态已刷新；已有索引和工作区保持不变。"
          : "连接仍未恢复；已有图片、对话、工作区和索引均已保留。",
        5200,
      );
      return result;
    } catch (error) {
      showError(error.message);
      return null;
    } finally {
      state.providerRevalidateInFlight = null;
      setBusy(button, false);
    }
  })();
  return state.providerRevalidateInFlight;
}
$("provider-revalidate").onclick = () => revalidateProvider({source: "manual"});
window.addEventListener("offline", () => {
  showToast("浏览器检测到网络可能已断开；当前工作区与已完成结果会保留。", 5200);
});
window.addEventListener("online", () => {
  showToast("网络可能已经恢复，正在执行一次有界连接验证。", 4200);
  if (state.onlineRecoveryRequested) return;
  state.onlineRecoveryRequested = true;
  revalidateProvider({source: "browser_online"});
});
$("provider-selection-form").onsubmit = async (event) => {
  event.preventDefault();
  const switchSequence = ++state.providerSwitchSequence;
  const mode = document.querySelector('input[name="provider-mode"]:checked')?.value;
  if (!mode) return showToast("请选择一种模型接入方式。");
  const cloudTier = document.querySelector('input[name="cloud-tier"]:checked')?.value || "standard";
  markRetrievalProjectionPending({
    mode,
    capabilities: {retrieval: false},
  });
  setBusy($("provider-save-selection"), true, "保存中");
  try {
    if (mode === "bailian") await saveUserCredentialIfNeeded();
    if (mode === "bailian" && state.provider?.mode !== "bailian") {
      $("provider-connection-result").textContent = "正在重新连接百炼云端模型……";
      $("health").textContent = "服务器映射 → 正在重新连接百炼云端模型……";
      showToast("正在创建全新的百炼连接并执行最小健康验证。", 4200);
    }
    const provider = await api("/providers/access/selection", {
      method: "PUT",
      body: JSON.stringify({
        mode,
        cloud_tier: cloudTier,
        credential_source: selectedCredentialSource(),
        region: "cn-beijing",
        api_host_override: $("provider-api-host").value || null,
        endpoint_mode: $("provider-endpoint-mode").value,
        workspace_id: $("provider-workspace-id").value || null,
      }),
    });
    if (switchSequence !== state.providerSwitchSequence) return;
    renderProviderAccess(provider);
    if (provider.mode === "bailian") await loadCloudIndexStatus();
    const projection = await refreshProviderAwareProjection({
      renderSystemCards: false,
    });
    if (!projection || switchSequence !== state.providerSwitchSequence) return;
    const coverage = projection.coverage;
    if (provider.connection_state === "READY" && coverage.missing_asset_ids.length) {
      await openIndexBackfill(
        "all_user_assets", null, null, {automatic: true},
      );
    } else if (
      provider.connection_state === "READY"
      && coverage.missing_asset_ids.length === 0
    ) {
      showToast("所有持久自定义图片均已具有当前模式的检索索引。");
    }
    if (!["local", "self_hosted"].includes(mode)) closeModelAccess();
    if (mode === "no_model") {
      showToast("已进入暂不接入模型模式；全部页面仍可浏览。");
    } else if ((provider.connection_state || provider.state) === "READY") {
      showToast(`已切换为${provider.mode_label}。`);
    } else {
      showToast("接入选择已保存；请继续完成该模式的连接或加载检查。", 4200);
    }
  } catch (error) {
    showError(error.message);
    await loadStatus({renderSystemCards: false}).catch(() => null);
  } finally {
    setBusy($("provider-save-selection"), false);
  }
};

function renderLocalPreflight(preflight) {
  const freeGiB = (Number(preflight.gpu?.free_vram_bytes || 0) / (1024 ** 3)).toFixed(1);
  const totalGiB = (Number(preflight.gpu?.total_vram_bytes || 0) / (1024 ** 3)).toFixed(1);
  const diagnostics = preflight.diagnostic_summary || {};
  const indexes = preflight.indexes || {};
  const preparedScopes = Object.entries(indexes.scopes || {})
    .filter(([, value]) => value?.status === "prepared")
    .map(([scope]) => scope);
  const dependencySummary = Object.entries(preflight.dependencies || {})
    .map(([name, value]) => `${name} ${value?.installed ? value.version || "已安装" : "缺失"}`)
    .join(" · ");
  $("local-provider-preflight-result").textContent =
    preflight.can_attempt ? "检查完成 · 可以尝试加载" : "检查完成 · 当前无法完整加载";
  $("local-provider-diagnostics").classList.remove("hidden");
  $("local-provider-diagnostics-content").innerHTML = [
    `<p><strong>CUDA：</strong>${preflight.cuda_available ? "可用" : "不可用"}</p>`,
    `<p><strong>GPU：</strong>${escapeHtml(preflight.gpu?.name || "未检测到")} · ${totalGiB} GiB</p>`,
    `<p><strong>当前可用显存：</strong>约 ${freeGiB} GiB</p>`,
    `<p><strong>VLM 权重：</strong>${preflight.weights?.vlm?.present ? "已找到" : "缺失"} · ${escapeHtml(preflight.weights?.vlm?.model_id || "")}</p>`,
    `<p><strong>Embedding 权重：</strong>${preflight.weights?.embedding?.present ? "已找到" : "缺失"} · ${escapeHtml(preflight.weights?.embedding?.model_id || "")}</p>`,
    `<p><strong>Embedding 推理源码：</strong>${preflight.weights?.embedding?.source_present ? "已找到" : "缺失"}</p>`,
    `<p><strong>模型清单：</strong>${preflight.manifest?.status === "ready" ? "已就绪" : "缺失或无效"} · ${escapeHtml(preflight.manifest?.relative_path || "")}</p>`,
    `<p><strong>标准索引：</strong>${indexes.all_prepared ? "产品、Train、Val 均已准备" : `已准备 ${preparedScopes.length}/3`} · 不会自动重编码</p>`,
    `<p><strong>本地推理依赖：</strong>${escapeHtml(dependencySummary || "尚未检查")}</p>`,
    `<p><strong>预计双模型峰值：</strong>约 ${Number(diagnostics.historical_dual_peak_gib || 12.62).toFixed(2)} GiB</p>`,
    `<p><strong>结论：</strong>${escapeHtml(preflight.conclusion || "状态待确认")}</p>`,
  ].join("");
  return preflight;
}

$("local-provider-preflight").onclick = async () => {
  setBusy($("local-provider-preflight"), true, "检查中");
  try {
    renderLocalPreflight(await api("/providers/access/local/preflight"));
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy($("local-provider-preflight"), false);
  }
};

$("local-provider-load").onclick = async () => {
  const switchSequence = state.providerSwitchSequence;
  setBusy($("local-provider-load"), true, "加载中");
  $("local-provider-load-result").textContent = "检查环境…";
  try {
    const result = await api("/providers/access/local/load", {
      method: "POST",
      body: JSON.stringify({
        force_low_vram_attempt: $("local-provider-force-low-vram").checked,
      }),
    });
    if (switchSequence !== state.providerSwitchSequence) return;
    renderLocalPreflight(result.preflight);
    renderProviderAccess(result.provider);
    await refreshProviderAwareProjection({renderSystemCards: false});
    if (!result.success) {
      $("local-provider-load-result").textContent = result.requires_confirmation
        ? "等待确认是否仍然尝试"
        : result.failure?.public_message || "本地模型加载未完成";
      $("local-provider-developer-details").classList.remove("hidden");
      $("local-provider-developer-code").textContent =
        `错误码：${result.failure?.code || "UNKNOWN"}`
        + `${result.failure?.reason_codes?.length ? ` · ${result.failure.reason_codes.join(" / ")}` : ""}`;
      showToast(result.failure?.public_message || "本地模型加载未完成。", 6500);
      return;
    }
    $("local-provider-load-result").textContent = result.provider.state === "READY"
      ? "VLM、Embedding 与索引均已就绪"
      : "模型已加载，检索索引尚未就绪";
    showToast("本地模型按需加载流程已完成。", 4800);
    await loadStatus();
  } catch (error) {
    $("local-provider-load-result").textContent = "加载失败，主后端仍可使用";
    showError(error.message);
  } finally {
    setBusy($("local-provider-load"), false);
  }
};

$("local-provider-unload").onclick = async () => {
  const switchSequence = state.providerSwitchSequence;
  setBusy($("local-provider-unload"), true, "卸载中");
  try {
    const result = await api("/providers/access/local/unload", {
      method: "POST",
      body: "{}",
    });
    if (switchSequence !== state.providerSwitchSequence) return;
    renderProviderAccess(result.provider);
    await refreshProviderAwareProjection({renderSystemCards: false});
    $("local-provider-load-result").textContent = "本地模型已卸载，显存清理已请求";
    showToast("本地模型已卸载；其他 Provider 与工作区状态不受影响。");
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy($("local-provider-unload"), false);
  }
};

$("self-hosted-provider-test").onclick = async () => {
  const switchSequence = state.providerSwitchSequence;
  setBusy($("self-hosted-provider-test"), true, "检查中");
  $("self-hosted-provider-result").textContent = "正在检查两个服务与本地隧道…";
  try {
    const result = await api("/providers/access/self-hosted/test", {
      method: "POST",
      body: "{}",
    });
    if (switchSequence !== state.providerSwitchSequence) return;
    renderProviderAccess(result.provider);
    await refreshProviderAwareProjection({renderSystemCards: false});
    $("self-hosted-provider-result").textContent = result.success
      ? `VLM ready；Embedding ready@${result.embedding.dimensions}；隧道 ready`
      : `${result.public_message} VLM：${result.vlm.status}；Embedding：${result.embedding.status}`;
    if (result.success) {
      showToast("服务器映射课程演示模式已就绪；未重启远端服务或隧道。", 4800);
    } else {
      showToast(result.public_message, 6000);
    }
  } catch (error) {
    $("self-hosted-provider-result").textContent = "检查失败";
    showError(error.message);
  } finally {
    setBusy($("self-hosted-provider-test"), false);
  }
};

$("provider-test-connection").onclick = async () => {
  const switchSequence = ++state.providerSwitchSequence;
  const cloudTier = document.querySelector('input[name="cloud-tier"]:checked')?.value || "standard";
  const source = selectedCredentialSource();
  setBusy($("provider-test-connection"), true, "测试中");
  $("provider-connection-result").textContent = "正在分别测试 VLM 与 Embedding…";
  try {
    await saveUserCredentialIfNeeded({required: true});
    await api("/providers/access/selection", {
      method: "PUT",
      body: JSON.stringify({
        mode: "bailian",
        cloud_tier: cloudTier,
        credential_source: source,
        region: "cn-beijing",
        api_host_override: $("provider-api-host").value || null,
        endpoint_mode: $("provider-endpoint-mode").value,
        workspace_id: $("provider-workspace-id").value || null,
      }),
    });
    if (switchSequence !== state.providerSwitchSequence) return;
    const result = await api("/providers/access/test", {
      method: "POST",
      body: JSON.stringify({
        cloud_tier: cloudTier,
        credential_source: source,
      }),
    });
    if (switchSequence !== state.providerSwitchSequence) return;
    renderProviderAccess(result.provider);
    await refreshProviderAwareProjection({renderSystemCards: false});
    const vlm = result.results?.vlm || {};
    const embedding = result.results?.embedding || {};
    const parts = [
      vlm.success ? `VLM 已连接（${vlm.model_id || result.provider.vlm?.model_id}）` : `VLM 失败（${vlm.code || "UNKNOWN"}）`,
      embedding.success ? `Embedding 已连接（${embedding.dimensions || 2560}维）` : `Embedding 失败（${embedding.code || "UNKNOWN"}）`,
    ];
    $("provider-connection-result").textContent = parts.join("；");
    if (!vlm.success || !embedding.success) {
      showToast(vlm.public_message || embedding.public_message || "连接测试部分失败，请检查脱敏错误码。", 6000);
    } else {
      showToast("百炼 VLM 与 2560 维 Embedding 连接测试通过。", 4200);
      await loadCloudIndexStatus();
    }
  } catch (error) {
    $("provider-api-key").value = "";
    $("provider-connection-result").textContent = "连接测试失败";
    showError(error.message);
  } finally {
    setBusy($("provider-test-connection"), false);
  }
};

$("cloud-index-build-first10").onclick = async () => {
  if ($("cloud-index-build-first10").disabled) {
    showToast("2369 项完整云索引已经建立，无需重复编码。");
    return;
  }
  setBusy($("cloud-index-build-first10"), true, "建立中");
  $("cloud-index-result").textContent = "正在按数字编号编码 0–9，共 10 张…";
  try {
    const result = await api("/providers/access/cloud-index/base", {
      method: "POST",
      body: JSON.stringify({target_base_items: 10}),
    });
    renderProviderAccess(result.provider);
    await loadStatus({renderSystemCards: false});
    if (!result.success) {
      $("cloud-index-result").textContent = `失败 · ${result.failure?.code || "UNKNOWN"}`;
      showToast(result.failure?.public_message || "云端索引建立失败，已有状态已保留。", 6500);
      return;
    }
    $("cloud-index-result").textContent = `已就绪 · ${result.index.items} 项 · ${result.index.dimensions} 维`;
    $("cloud-index-scope").textContent = result.scope_message;
    $("cloud-retrieval-scope").textContent = `${result.scope_message}。未入索引图片仍可浏览、选中并直接发送给云端 VLM，但不会出现在检索结果中。`;
    showToast("Train 数字编号 0–9 的独立 2560 维云端索引已建立。", 4800);
    await loadStatus();
  } catch (error) {
    $("cloud-index-result").textContent = "建立失败";
    showError(error.message);
  } finally {
    setBusy($("cloud-index-build-first10"), false);
  }
};

async function loadStatus({renderSystemCards = true} = {}) {
  const projectionSequence = ++state.retrievalProjectionSequence;
  try {
    const [health, status] = await Promise.all([api("/health"), api("/system/status")]);
    if (projectionSequence !== state.retrievalProjectionSequence) return null;
    if (health.provider) renderProviderAccess(health.provider);
    if (renderSystemCards) {
      const cards = [
        ["API", status.api],
        ["VLM", status.vlm.status],
        ["检索", `${status.retriever_label} · ${status.retrieval.items || 0} items`],
        ["检索 revision", status.retrieval.embedding?.model_revision || "-"],
        ["检索 index", status.retrieval.index_version || "-"],
        ["会话", status.product.vqa_sessions],
        ["冻结基线", status.frozen_prompt_suite],
        ["课程 Candidate", status.course_prompt_candidate?.prompt_id],
        ["默认 Chat", status.multiturn_chat_candidate?.prompt_id],
        ["多图内容", status.multi_image_content_candidate?.prompt_id],
        ["训练", status.training ? "RUNNING" : "OFF"],
        ["Val/Test/Blind", status.val_test_blind_read ? "READ" : "UNREAD"],
      ];
      $("system-status").innerHTML = cards.map(([label, value]) => `<div class="status-card"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`).join("");
    }
    const projection = renderRetrievalRuntimeStatus(status);
    $("prompt-state").textContent = status.multiturn_chat_candidate?.default_chat_route
      ? "Suite V1 frozen · Chat V2 · Phase 5.4 Router/Profile Candidates"
      : "Suite V1 frozen · Chat V2 candidate · Phase 5.4 Router/Profile Candidates";
    return {health, status, projection};
  } catch (error) {
    if (projectionSequence === state.retrievalProjectionSequence) showError(error.message);
    return null;
  }
}

async function loadHistory() {
  try {
    const [history, tasks] = await Promise.all([api("/history"), api("/tasks")]);
    const items = [
      ...tasks.items.map((item) => ({task_type: item.task_type, status: item.status, started_at: item.created_at, request_id: item.task_id})),
      ...history.items,
    ].slice(0, 40);
    $("history-result").innerHTML = items.length
      ? items.map((item) => `<div class="history-item"><strong>${escapeHtml(item.task_type || "request")}</strong>${escapeHtml(item.status || "-")}<small>${escapeHtml(item.started_at || item.created_at || "")} · ${escapeHtml(item.request_id || "")}</small></div>`).join("")
      : '<span class="empty-context">暂无记录</span>';
  } catch (error) { showError(error.message); }
}

$("load-history").onclick = loadHistory;
$("refresh").onclick = async () => {
  showError("");
  try {
    await Promise.all([loadProviderAccess(), loadConversations(false)]);
    await refreshProviderAwareProjection();
  } catch (error) { showError(error.message); }
};

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.onclick = async () => {
    const node = $(button.dataset.copy);
    if (!node) return;
    await navigator.clipboard.writeText(node.innerText);
    const original = button.textContent;
    button.textContent = "已复制";
    setTimeout(() => { button.textContent = original; }, 1200);
  };
});

function coverageQuery(scope, assetId = null, libraryId = null) {
  const query = new URLSearchParams({scope});
  if (assetId) query.set("asset_id", assetId);
  if (libraryId) query.set("library_id", libraryId);
  return `/provider-index/coverage?${query}`;
}

async function loadProviderIndexCoverage({providerProjectionSequence = null} = {}) {
  const coverage = await api(coverageQuery("all_user_assets"));
  if (
    providerProjectionSequence !== null
    && providerProjectionSequence !== state.providerProjectionSequence
  ) return null;
  state.providerIndexCoverage = coverage;
  const counts = coverage.counts;
  $("provider-index-coverage-summary").textContent =
    `${coverage.identity.mode_label} · 持久用户资产 ${counts.total} · `
    + `已索引 ${counts.indexed} · 待补齐 ${counts.pending + counts.identity_mismatch} · `
    + `失败 ${counts.failed}`;
  return coverage;
}

async function refreshProviderAwareProjection({renderSystemCards = true} = {}) {
  const projectionSequence = ++state.providerProjectionSequence;
  const statusProjection = await loadStatus({renderSystemCards});
  if (
    projectionSequence !== state.providerProjectionSequence
    || !statusProjection
  ) return null;
  const [, coverage] = await Promise.all([
    loadAssets({providerProjectionSequence: projectionSequence}),
    loadProviderIndexCoverage({
      providerProjectionSequence: projectionSequence,
    }),
  ]);
  if (
    projectionSequence !== state.providerProjectionSequence
    || !coverage
  ) return null;
  renderContext();
  if (state.selectedRef) await selectAsset(state.selectedRef);
  if (projectionSequence !== state.providerProjectionSequence) return null;
  await loadFunctionWorkspaces();
  if (projectionSequence !== state.providerProjectionSequence) return null;
  return {
    status: statusProjection,
    coverage,
    provider: state.provider,
  };
}

function newBackfillOperationId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `backfill-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function setBackfillActionBusy(busy) {
  ["backfill-current-library", "backfill-all-assets"].forEach((id) => {
    const button = $(id);
    if (button) {
      button.disabled = busy;
      button.setAttribute("aria-busy", String(Boolean(busy)));
    }
  });
}

function renderBackfillDialog(coverage, {preserveProgress = false} = {}) {
  const counts = coverage.counts;
  const identity = coverage.identity;
  const confirm = coverage.confirmation;
  const missing = coverage.missing_asset_ids.length;
  $("index-backfill-title").textContent = `${identity.mode_label}索引补齐`;
  $("index-backfill-message").textContent = !identity.ready
    ? "当前 Embedding Provider 尚未就绪，请先完成模型连接验证。"
    : missing
      ? `已发现 ${missing} 张持久自定义图片待补齐。确认前不会发送图片或产生 API 费用。`
      : "当前范围内所有持久图片均已具备当前模式的检索索引，无需补齐。";
  const cost = confirm.estimated_cost_cny;
  const credential = {
    user_session: "用户自己的 API Key",
    course_default: "课程演示默认 API Key",
  }[confirm.credential_source] || "不使用百炼凭据";
  $("index-backfill-details").innerHTML = [
    ["当前 Provider", identity.mode_label],
    ["当前 Embedding", `${identity.model_id}@${identity.dimension}`],
    ["处理范围", {
      asset: "当前图片",
      library: "当前图库的持久用户资产",
      all_user_assets: "全部持久自定义资产",
    }[coverage.scope] || coverage.scope],
    ["目标总数", counts.total],
    ["已完成", counts.indexed],
    ["待补齐", missing],
    ["失败", counts.failed],
    ["处理位置", confirm.destination],
    ["凭据来源", credential],
    ["数据说明", confirm.external_processing
      ? "确认后，目标图片会发送至阿里云百炼进行 Embedding 推理。"
      : "在当前本地或服务器 Embedding 环境处理，不产生百炼 API 费用。"],
    ["费用估算", cost
      ? `约 ¥${Number(cost.minimum).toFixed(4)}–¥${Number(cost.maximum).toFixed(4)}；${cost.basis}`
      : "无百炼 API 费用"],
    ["预计耗时", `${confirm.estimated_seconds.minimum}–${confirm.estimated_seconds.maximum} 秒`],
    ["幂等与恢复", "已成功项跳过；最多64张/任务；支持取消和断点续跑"],
  ].map(([key, value]) => `<dt>${escapeHtml(String(key))}</dt><dd>${escapeHtml(String(value))}</dd>`).join("");
  $("index-backfill-technical-details").innerHTML = [
    ["Provider Identity", String(identity.identity_sha256 || "").slice(0, 12)],
    ["操作 ID", state.backfillOperationId || "尚未创建"],
    ["任务 ID", state.activeBackfillTaskId || "尚未创建"],
  ].map(([key, value]) => `<dt>${escapeHtml(String(key))}</dt><dd>${escapeHtml(String(value))}</dd>`).join("");
  $("index-backfill-confirm").classList.toggle("hidden", !missing || !identity.ready);
  $("index-backfill-later").textContent = missing ? "稍后处理" : "关闭";
  if (!preserveProgress) {
    $("index-backfill-progress").classList.add("hidden");
    $("index-backfill-cancel-task").classList.add("hidden");
  }
}

async function openIndexBackfill(
  scope,
  assetId = null,
  libraryId = null,
  {automatic = false, triggerButton = null} = {},
) {
  if (state.backfillScanInProgress) {
    showToast("正在扫描当前 Provider 索引覆盖，请稍候。");
    return null;
  }
  state.backfillScanInProgress = true;
  state.backfillReturnFocus = triggerButton || document.activeElement;
  state.backfillOperationId = newBackfillOperationId();
  state.pendingBackfillScope = {scope, asset_id: assetId, library_id: libraryId};
  if (triggerButton) setBusy(triggerButton, true, "正在扫描");
  try {
    const coverage = await api(coverageQuery(scope, assetId, libraryId));
    renderBackfillDialog(coverage);
    const dialog = $("index-backfill-dialog");
    if (!dialog.open) dialog.showModal();
    queueMicrotask(() => {
      const target = !$("index-backfill-confirm").classList.contains("hidden")
        ? $("index-backfill-confirm")
        : $("index-backfill-close");
      target?.focus();
    });
    if (!coverage.identity.ready) {
      showToast("当前 Embedding Provider 尚未就绪，请先完成模型连接验证。", 6000);
    } else if (!coverage.missing_asset_ids.length) {
      showToast("当前范围内所有持久图片均已具备当前模式的检索索引，无需补齐。", 5200);
    } else if (automatic) {
      showToast(`检测到 ${coverage.missing_asset_ids.length} 张持久图片待补齐；不会自动外发。`, 6000);
    } else {
      showToast(`已发现 ${coverage.missing_asset_ids.length} 张待补齐，等待用户确认。`, 4200);
    }
    return coverage;
  } finally {
    state.backfillScanInProgress = false;
    if (triggerButton) setBusy(triggerButton, false);
  }
}

function renderBackfillTask(task) {
  const progress = task.progress || {};
  const node = $("index-backfill-progress");
  node.classList.remove("hidden");
  const statusLabel = {
    PENDING: "等待开始",
    RUNNING: "正在补齐",
    ENCODING: "正在补齐",
    SUCCESS: "已完成",
    CANCELLED: "已取消",
    BILLING_STOPPED: "Billing 停止",
    HARD_FAILED: "Provider 失败",
    IDENTITY_MISMATCH: "Provider 身份已变化",
  }[task.status] || task.status;
  node.textContent =
    `当前 Provider 索引补齐：${progress.completed || 0} / ${progress.total || 0}\n`
    + `跳过 ${progress.skipped || 0} · 失败 ${progress.failed || 0} · 剩余 ${progress.remaining || 0}`
    + `${task.current_asset_id ? `\n当前图片：${task.current_asset_id}` : ""}`
    + `\n状态：${statusLabel}`;
  const running = ["PENDING", "RUNNING", "ENCODING"].includes(task.status);
  $("index-backfill-confirm").classList.toggle("hidden", running);
  $("index-backfill-cancel-task").classList.toggle(
    "hidden", !running,
  );
  setBackfillActionBusy(running);
  $("index-backfill-technical-details").innerHTML = [
    ["操作 ID", task.operation_id || state.backfillOperationId || "-"],
    ["任务 ID", task.task_id || "-"],
    ["Provider Identity", String(task.provider_identity_sha256 || "").slice(0, 12)],
  ].map(([key, value]) => `<dt>${escapeHtml(String(key))}</dt><dd>${escapeHtml(String(value))}</dd>`).join("");
}

async function pollBackfillTask(taskId) {
  clearTimeout(state.backfillPollTimer);
  const task = await api(`/provider-index/tasks/${encodeURIComponent(taskId)}`);
  renderBackfillTask(task);
  if (["SUCCESS", "CANCELLED", "BILLING_STOPPED", "HARD_FAILED", "IDENTITY_MISMATCH"].includes(task.status)) {
    state.activeBackfillTaskId = null;
    setBackfillActionBusy(false);
    await loadAssets();
    const scope = state.pendingBackfillScope || {scope: "all_user_assets"};
    const [coverage] = await Promise.all([
      api(coverageQuery(scope.scope, scope.asset_id, scope.library_id)),
      loadProviderIndexCoverage(),
    ]);
    renderBackfillDialog(coverage, {preserveProgress: true});
    renderBackfillTask(task);
    $("index-backfill-confirm").classList.add("hidden");
    if (task.public_message) showToast(task.public_message, 7000);
    else if (task.status === "SUCCESS") showToast("索引补齐已完成，覆盖状态已更新。", 5200);
    return task;
  }
  state.backfillPollTimer = setTimeout(() => {
    pollBackfillTask(taskId).catch((error) => showError(error.message));
  }, 500);
  return task;
}

$("index-backfill-confirm").onclick = async () => {
  if (!state.pendingBackfillScope) return;
  setBusy($("index-backfill-confirm"), true, "启动中");
  try {
    const result = await api("/provider-index/tasks", {
      method: "POST",
      body: JSON.stringify({
        ...state.pendingBackfillScope,
        confirmed_by_user: true,
        operation_id: state.backfillOperationId,
      }),
    });
    if (!result.task) {
      showToast(result.message);
      return;
    }
    state.activeBackfillTaskId = result.task_id || result.task.task_id;
    renderBackfillTask(result.task);
    await pollBackfillTask(state.activeBackfillTaskId);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy($("index-backfill-confirm"), false);
  }
};
$("index-backfill-cancel-task").onclick = async () => {
  if (!state.activeBackfillTaskId) return;
  const task = await api(`/provider-index/tasks/${encodeURIComponent(state.activeBackfillTaskId)}/cancel`, {
    method: "POST", body: "{}",
  });
  renderBackfillTask(task);
};
$("backfill-selected-asset").onclick = () => {
  const asset = findAsset(state.selectedRef);
  if (!asset || asset.source !== "local") {
    return showToast("当前图片不是持久自定义资产，无需使用此补齐入口。");
  }
  return openIndexBackfill("asset", asset.sourceAssetId)
    .catch((error) => showError(error.message));
};
document.addEventListener("click", (event) => {
  const backfillButton = event.target.closest?.("[data-backfill-action]");
  if (backfillButton) {
    event.preventDefault();
    const scope = backfillButton.dataset.backfillAction;
    const libraryId = scope === "library" ? $("library-select").value : null;
    openIndexBackfill(scope, null, libraryId, {triggerButton: backfillButton})
      .catch((error) => showError(error.message));
    return;
  }
  const button = event.target.closest?.("[data-index-asset]");
  if (!button) return;
  event.preventDefault();
  event.stopPropagation();
  openIndexBackfill("asset", button.dataset.indexAsset)
    .catch((error) => showError(error.message));
});

$("index-backfill-dialog").addEventListener("close", () => {
  const target = state.backfillReturnFocus;
  if (target && document.contains(target)) target.focus();
  state.backfillReturnFocus = null;
});

async function init() {
  try {
    await loadProviderAccess();
    await loadContentLengthProfiles();
    await loadLibraries();
    await Promise.all([loadAssets(), loadConversations(false), loadStatus(), loadCanonicalPreview()]);
    await loadProviderIndexCoverage();
    await loadFunctionWorkspaces();
    const latest = state.conversations[0];
    if (latest) await switchConversation(latest.conversation_id);
  } catch (error) {
    showError(error.message);
  }
}

init();
