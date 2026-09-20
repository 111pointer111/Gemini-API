const TERMINAL_JOB_STATES = new Set(["completed", "partial", "failed", "cancelled"]);

const modeCopy = {
  chat: { eyebrow: "CONVERSATION", title: "和 Gemini 对话" },
  image: { eyebrow: "IMAGE STUDIO", title: "批量生成原图" },
  video: { eyebrow: "VIDEO STUDIO", title: "把画面变成视频" },
  audio: { eyebrow: "AUDIO STUDIO", title: "生成声音与音乐" },
  library: { eyebrow: "LOCAL ARCHIVE", title: "本地资源库" },
};

const kindCopy = {
  image: {
    mark: "IMG",
    title: "生成图片",
    result: "IMAGE RUNS",
    empty: "描述画面，Gemini 会把原图保存在本机。",
    placeholder: "雨夜中的上海街巷，电影感灯光，路面积水映出霓虹…",
    note: "每张图独立生成，最多同时处理 2 个请求。",
  },
  video: {
    mark: "VID",
    title: "生成视频",
    result: "VIDEO RUNS",
    empty: "视频生成时间较长，页面会自动轮询并保存完成文件。",
    placeholder: "清晨的海岸，低机位缓慢向前移动，海雾掠过黑色礁石…",
    note: "视频会在后台持续轮询，切换页面不会中断任务。",
  },
  audio: {
    mark: "AUD",
    title: "生成音频",
    result: "AUDIO RUNS",
    empty: "描述音乐、氛围或声音，完成后可直接试听和下载。",
    placeholder: "20 秒的氛围电子音乐，温暖模拟合成器，缓慢渐入，无人声…",
    note: "音频任务一次生成 1 个结果，完成后保存原始音频。",
  },
};

const state = {
  sessionId: null,
  mode: "chat",
  kind: "image",
  chatBusy: false,
  batch: 1,
  aspectRatio: "auto",
  assets: [],
  selected: new Set(),
  activeJobs: new Map(),
  libraryFilter: "all",
  viewerAsset: null,
};

const elements = {
  modeButtons: [...document.querySelectorAll(".mode-button")],
  chatView: document.querySelector("#chat-view"),
  studioView: document.querySelector("#studio-view"),
  libraryView: document.querySelector("#library-view"),
  viewEyebrow: document.querySelector("#view-eyebrow"),
  viewTitle: document.querySelector("#view-title"),
  resetChat: document.querySelector("#reset-chat"),
  modelSelect: document.querySelector("#model-select"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  conversation: document.querySelector("#conversation"),
  chatForm: document.querySelector("#chat-form"),
  chatInput: document.querySelector("#chat-input"),
  generationForm: document.querySelector("#generation-form"),
  generationPrompt: document.querySelector("#generation-prompt"),
  kindMark: document.querySelector("#kind-mark"),
  formTitle: document.querySelector("#form-title"),
  resultKicker: document.querySelector("#result-kicker"),
  emptyCopy: document.querySelector("#empty-copy"),
  runNote: document.querySelector("#run-note"),
  aspectControl: document.querySelector("#aspect-control"),
  aspectOptions: [...document.querySelectorAll("#aspect-options button")],
  batchControl: document.querySelector("#batch-control"),
  batchDown: document.querySelector("#batch-down"),
  batchUp: document.querySelector("#batch-up"),
  batchValue: document.querySelector("#batch-value"),
  generateLabel: document.querySelector("#generate-label"),
  gallery: document.querySelector("#gallery"),
  jobIndicator: document.querySelector("#job-indicator"),
  jobCopy: document.querySelector("#job-copy"),
  libraryGrid: document.querySelector("#library-grid"),
  libraryFilters: [...document.querySelectorAll(".library-filters button")],
  selectionBar: document.querySelector("#selection-bar"),
  selectionCount: document.querySelector("#selection-count"),
  clearSelection: document.querySelector("#clear-selection"),
  downloadSelection: document.querySelector("#download-selection"),
  viewer: document.querySelector("#viewer"),
  viewerClose: document.querySelector("#viewer-close"),
  viewerMedia: document.querySelector("#viewer-media"),
  viewerKind: document.querySelector("#viewer-kind"),
  viewerPrompt: document.querySelector("#viewer-prompt"),
  viewerModel: document.querySelector("#viewer-model"),
  viewerTime: document.querySelector("#viewer-time"),
  viewerDownload: document.querySelector("#viewer-download"),
  reusePrompt: document.querySelector("#reuse-prompt"),
  toast: document.querySelector("#toast"),
};

function showToast(message, type = "info") {
  elements.toast.textContent = message;
  elements.toast.classList.toggle("is-error", type === "error");
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    elements.toast.classList.remove("is-visible");
  }, 4300);
}

function errorMessage(body, status) {
  if (typeof body?.detail === "string") return body.detail;
  if (Array.isArray(body?.detail)) {
    return body.detail.map((item) => item?.msg ?? "参数不正确").join("；");
  }
  return `请求失败（${status}）`;
}

async function requestJson(url, options = {}) {
  const headers = options.body ? { "Content-Type": "application/json", ...(options.headers ?? {}) } : options.headers;
  const response = await fetch(url, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, response.status));
  return body;
}

function updateModelOptions(models) {
  if (!models?.length) return;
  const current = elements.modelSelect.value;
  elements.modelSelect.replaceChildren(
    ...models.map((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.label || model.name;
      return option;
    }),
  );
  const preferred = models.find((model) => model.name === current)
    ?? models.find((model) => model.name.includes("pro"))
    ?? models[0];
  elements.modelSelect.value = preferred.name;
}

async function loadStatus() {
  try {
    const status = await requestJson("/api/status");
    elements.statusDot.classList.add("is-ready");
    elements.statusText.textContent = `${status.account_status} · ${status.cookie_browser ?? "Browser"}`;
    updateModelOptions(status.models);
  } catch (error) {
    elements.statusDot.classList.add("is-error");
    elements.statusText.textContent = "连接失败";
    showToast(error.message, "error");
  }
}

async function loadAssets({ quiet = false } = {}) {
  try {
    const result = await requestJson("/api/assets");
    state.assets = result.assets ?? [];
    renderCurrentSurface();
  } catch (error) {
    if (!quiet) showToast(error.message, "error");
  }
}

function setMode(mode) {
  const copy = modeCopy[mode];
  if (!copy) return;
  state.mode = mode;
  if (["image", "video", "audio"].includes(mode)) state.kind = mode;

  elements.chatView.classList.toggle("is-active", mode === "chat");
  elements.studioView.classList.toggle("is-active", ["image", "video", "audio"].includes(mode));
  elements.libraryView.classList.toggle("is-active", mode === "library");
  elements.resetChat.hidden = mode !== "chat";
  elements.viewEyebrow.textContent = copy.eyebrow;
  elements.viewTitle.textContent = copy.title;
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });

  if (["image", "video", "audio"].includes(mode)) {
    updateGenerationControls();
    renderGallery();
    window.requestAnimationFrame(() => elements.generationPrompt.focus());
  } else if (mode === "library") {
    renderLibrary();
  } else {
    window.requestAnimationFrame(() => elements.chatInput.focus());
  }
}

function updateGenerationControls() {
  const copy = kindCopy[state.kind];
  const isImage = state.kind === "image";
  elements.kindMark.textContent = copy.mark;
  elements.formTitle.textContent = copy.title;
  elements.resultKicker.textContent = copy.result;
  elements.emptyCopy.textContent = copy.empty;
  elements.generationPrompt.placeholder = copy.placeholder;
  elements.runNote.textContent = copy.note;
  elements.aspectControl.hidden = !isImage;
  elements.batchControl.hidden = !isImage;
  elements.generateLabel.textContent = isImage ? `生成 ${state.batch} 张图片` : copy.title;
  elements.batchValue.textContent = String(state.batch);
}

function appendMessage(role, text, images = []) {
  document.querySelector("#chat-intro")?.remove();
  const message = document.createElement("article");
  message.className = `message is-${role}`;

  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "user" ? "YOU" : "GEMINI";

  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = text;

  if (images.length) {
    const imageGrid = document.createElement("div");
    imageGrid.className = "message-images";
    images.forEach((url) => {
      const image = document.createElement("img");
      image.src = url;
      image.alt = "Gemini 返回的图片";
      image.loading = "lazy";
      imageGrid.append(image);
    });
    body.append(imageGrid);
  }

  message.append(label, body);
  elements.conversation.append(message);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  return message;
}

function resetChat() {
  state.sessionId = null;
  elements.conversation.replaceChildren();
  const intro = document.createElement("div");
  intro.className = "intro-copy";
  intro.id = "chat-intro";
  const mark = document.createElement("span");
  mark.className = "intro-mark";
  mark.textContent = "✦";
  const title = document.createElement("h3");
  title.textContent = "新对话已准备";
  const copy = document.createElement("p");
  copy.textContent = "下一条消息不会继承上一段对话的上下文。";
  intro.append(mark, title, copy);
  elements.conversation.append(intro);
  elements.chatInput.focus();
}

async function submitChat(event) {
  event.preventDefault();
  const message = elements.chatInput.value.trim();
  if (!message || state.chatBusy) return;

  appendMessage("user", message);
  elements.chatInput.value = "";
  elements.chatInput.style.height = "auto";
  const pending = appendMessage("assistant", "正在思考");
  pending.classList.add("is-pending");
  state.chatBusy = true;
  elements.chatForm.querySelector("button[type='submit']").disabled = true;

  try {
    const result = await requestJson("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        session_id: state.sessionId,
        model: elements.modelSelect.value,
      }),
    });
    state.sessionId = result.session_id;
    pending.remove();
    appendMessage("assistant", result.text, result.images);
  } catch (error) {
    pending.remove();
    showToast(error.message, "error");
  } finally {
    state.chatBusy = false;
    elements.chatForm.querySelector("button[type='submit']").disabled = false;
    elements.chatInput.focus();
  }
}

function activeJobsFor(kind) {
  return [...state.activeJobs.values()].filter((job) => job.kind === kind);
}

function pendingCount(kind) {
  return activeJobsFor(kind).reduce((total, job) => total + Math.max(0, job.count - job.completed), 0);
}

function mediaElement(asset, { detailed = false } = {}) {
  if (asset.kind === "image") {
    const image = document.createElement("img");
    image.src = asset.url;
    image.alt = asset.prompt;
    image.loading = detailed ? "eager" : "lazy";
    return image;
  }
  if (asset.kind === "video") {
    const video = document.createElement("video");
    video.src = asset.url;
    video.poster = asset.preview_url ?? "";
    video.preload = "metadata";
    video.playsInline = true;
    video.controls = detailed;
    video.muted = !detailed;
    return video;
  }
  if (detailed) {
    const audio = document.createElement("audio");
    audio.src = asset.url;
    audio.controls = true;
    return audio;
  }
  const art = document.createElement("div");
  art.className = "audio-art";
  art.innerHTML = '<svg viewBox="0 0 64 64" aria-hidden="true"><path d="M15 35v-6M23 43V21M31 48V16M39 42V22M47 35v-6" /></svg>';
  return art;
}

function formattedTime(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "未知时间"
    : new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

function createAssetCard(asset) {
  const card = document.createElement("article");
  card.className = "asset-card";
  card.dataset.assetId = asset.id;
  card.classList.toggle("is-selected", state.selected.has(asset.id));

  const preview = document.createElement("button");
  preview.className = "asset-preview";
  preview.type = "button";
  preview.setAttribute("aria-label", `查看${asset.kind === "image" ? "原图" : "媒体"}`);
  const media = mediaElement(asset);
  preview.append(media);
  preview.addEventListener("click", () => openViewer(asset));
  if (asset.kind === "video") {
    preview.addEventListener("mouseenter", () => media.play().catch(() => {}));
    preview.addEventListener("mouseleave", () => {
      media.pause();
      media.currentTime = 0;
    });
  }

  const select = document.createElement("button");
  select.className = "asset-select";
  select.type = "button";
  select.textContent = state.selected.has(asset.id) ? "✓" : "+";
  select.setAttribute("aria-label", state.selected.has(asset.id) ? "取消选择" : "选择资源");
  select.addEventListener("click", () => toggleSelection(asset.id));

  const download = document.createElement("a");
  download.className = "asset-download";
  download.href = asset.download_url;
  download.textContent = "↓";
  download.setAttribute("aria-label", asset.source_url ? "打开 Gemini 原图" : "下载本地文件");
  if (asset.source_url) {
    download.target = "_blank";
    download.rel = "noopener";
  }

  const meta = document.createElement("div");
  meta.className = "asset-meta";
  const prompt = document.createElement("p");
  prompt.textContent = asset.prompt;
  const details = document.createElement("span");
  details.textContent = `${asset.model} · ${formattedTime(asset.created_at)}`;
  meta.append(prompt, details);
  card.append(preview, select, download, meta);
  return card;
}

function createSkeleton() {
  const skeleton = document.createElement("div");
  skeleton.className = "skeleton-card";
  skeleton.setAttribute("aria-label", "生成中");
  return skeleton;
}

function renderGallery() {
  const assets = state.assets.filter((asset) => asset.kind === state.kind);
  const skeletonCount = pendingCount(state.kind);
  const grid = document.createElement("div");
  grid.className = "gallery-grid";
  for (let index = 0; index < skeletonCount; index += 1) grid.append(createSkeleton());
  assets.forEach((asset) => grid.append(createAssetCard(asset)));

  if (grid.childElementCount) {
    elements.gallery.replaceChildren(grid);
  } else {
    const empty = document.createElement("div");
    empty.className = "gallery-empty";
    const visual = document.createElement("div");
    visual.className = "empty-visual";
    visual.append(document.createElement("span"));
    const title = document.createElement("h3");
    title.textContent = "还没有作品";
    const copy = document.createElement("p");
    copy.textContent = kindCopy[state.kind].empty;
    empty.append(visual, title, copy);
    elements.gallery.replaceChildren(empty);
  }

  const jobs = activeJobsFor(state.kind);
  elements.jobIndicator.hidden = jobs.length === 0;
  if (jobs.length) {
    elements.jobCopy.textContent = `${jobs.length} 个任务 · ${skeletonCount} 个结果处理中`;
  }
}

function renderLibrary() {
  const assets = state.libraryFilter === "all"
    ? state.assets
    : state.assets.filter((asset) => asset.kind === state.libraryFilter);
  if (!assets.length) {
    const empty = document.createElement("div");
    empty.className = "library-empty";
    empty.textContent = "这里还没有对应的生成记录。";
    elements.libraryGrid.replaceChildren(empty);
    return;
  }
  elements.libraryGrid.replaceChildren(...assets.map(createAssetCard));
}

function renderCurrentSurface() {
  if (["image", "video", "audio"].includes(state.mode)) renderGallery();
  if (state.mode === "library") renderLibrary();
  updateSelectionBar();
}

function mergeAssets(incoming) {
  const byId = new Map(state.assets.map((asset) => [asset.id, asset]));
  incoming.forEach((asset) => byId.set(asset.id, asset));
  state.assets = [...byId.values()].sort((left, right) => right.created_at.localeCompare(left.created_at));
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function pollJob(jobId) {
  while (state.activeJobs.has(jobId)) {
    await delay(1600);
    try {
      const job = await requestJson(`/api/jobs/${jobId}`);
      state.activeJobs.set(jobId, job);
      mergeAssets(job.assets ?? []);
      renderCurrentSurface();
      if (TERMINAL_JOB_STATES.has(job.status)) {
        state.activeJobs.delete(jobId);
        renderCurrentSurface();
        if (job.status === "completed") {
          showToast(job.kind === "image" ? `${job.completed} 张原图已保存` : "生成完成，原文件已保存");
        } else {
          showToast(job.error || `任务状态：${job.status}`, "error");
        }
        await loadAssets({ quiet: true });
        return;
      }
    } catch (error) {
      state.activeJobs.delete(jobId);
      renderCurrentSurface();
      showToast(error.message, "error");
      return;
    }
  }
}

async function submitGeneration(event) {
  event.preventDefault();
  const prompt = elements.generationPrompt.value.trim();
  if (!prompt) return;
  const button = elements.generationForm.querySelector("button[type='submit']");
  button.disabled = true;
  try {
    const job = await requestJson("/api/generations", {
      method: "POST",
      body: JSON.stringify({
        kind: state.kind,
        prompt,
        model: elements.modelSelect.value,
        count: state.kind === "image" ? state.batch : 1,
        aspect_ratio: state.kind === "image" ? state.aspectRatio : "auto",
      }),
    });
    state.activeJobs.set(job.id, job);
    renderGallery();
    showToast("任务已加入队列，可以继续创建下一项。 ");
    void pollJob(job.id);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function toggleSelection(assetId) {
  if (state.selected.has(assetId)) state.selected.delete(assetId);
  else state.selected.add(assetId);
  renderCurrentSurface();
}

function updateSelectionBar() {
  elements.selectionBar.hidden = state.selected.size === 0;
  elements.selectionCount.textContent = String(state.selected.size);
}

function clearSelection() {
  state.selected.clear();
  renderCurrentSurface();
}

async function downloadSelection() {
  if (!state.selected.size) return;
  elements.downloadSelection.disabled = true;
  try {
    const response = await fetch("/api/assets/download-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset_ids: [...state.selected] }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(errorMessage(body, response.status));
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "gemini-assets.zip";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("已打包所选原文件");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    elements.downloadSelection.disabled = false;
  }
}

function openViewer(asset) {
  state.viewerAsset = asset;
  elements.viewerKind.textContent = asset.kind.toUpperCase();
  elements.viewerPrompt.textContent = asset.prompt;
  elements.viewerModel.textContent = asset.model;
  elements.viewerTime.textContent = formattedTime(asset.created_at);
  elements.viewerDownload.href = asset.download_url;
  elements.viewerDownload.textContent = asset.source_url ? "打开 Gemini 原图" : "下载原文件";
  elements.viewerDownload.target = asset.source_url ? "_blank" : "";
  elements.viewerDownload.rel = asset.source_url ? "noopener" : "";
  elements.viewerMedia.replaceChildren(mediaElement(asset, { detailed: true }));
  elements.viewer.showModal();
}

function reuseViewerPrompt() {
  const asset = state.viewerAsset;
  if (!asset) return;
  elements.viewer.close();
  setMode(asset.kind);
  elements.generationPrompt.value = asset.prompt;
  elements.generationPrompt.focus();
  showToast("提示词已放回编辑器");
}

elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

elements.resetChat.addEventListener("click", resetChat);
elements.chatForm.addEventListener("submit", submitChat);
elements.generationForm.addEventListener("submit", submitGeneration);
elements.clearSelection.addEventListener("click", clearSelection);
elements.downloadSelection.addEventListener("click", downloadSelection);
elements.viewerClose.addEventListener("click", () => elements.viewer.close());
elements.reusePrompt.addEventListener("click", reuseViewerPrompt);

elements.viewer.addEventListener("click", (event) => {
  if (event.target === elements.viewer) elements.viewer.close();
});

elements.modelSelect.addEventListener("change", () => {
  state.sessionId = null;
  showToast("模型已切换，新消息会开始独立对话。 ");
});

elements.chatInput.addEventListener("input", () => {
  elements.chatInput.style.height = "auto";
  elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 170)}px`;
});

elements.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-starter]").forEach((button) => {
  button.addEventListener("click", () => {
    elements.chatInput.value = button.dataset.starter ?? "";
    elements.chatInput.focus();
  });
});

elements.aspectOptions.forEach((button) => {
  button.addEventListener("click", () => {
    state.aspectRatio = button.dataset.value ?? "auto";
    elements.aspectOptions.forEach((item) => item.classList.toggle("is-active", item === button));
  });
});

elements.batchDown.addEventListener("click", () => {
  state.batch = Math.max(1, state.batch - 1);
  updateGenerationControls();
});

elements.batchUp.addEventListener("click", () => {
  state.batch = Math.min(4, state.batch + 1);
  updateGenerationControls();
});

elements.libraryFilters.forEach((button) => {
  button.addEventListener("click", () => {
    state.libraryFilter = button.dataset.filter ?? "all";
    elements.libraryFilters.forEach((item) => item.classList.toggle("is-active", item === button));
    renderLibrary();
  });
});

setMode("chat");
await Promise.all([loadStatus(), loadAssets({ quiet: true })]);
