const state = {
  sessionId: null,
  mode: "chat",
  busy: false,
};

const elements = {
  modeButtons: [...document.querySelectorAll(".mode-button")],
  chatView: document.querySelector("#chat-view"),
  imageView: document.querySelector("#image-view"),
  viewEyebrow: document.querySelector("#view-eyebrow"),
  viewTitle: document.querySelector("#view-title"),
  resetChat: document.querySelector("#reset-chat"),
  modelSelect: document.querySelector("#model-select"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  conversation: document.querySelector("#conversation"),
  chatForm: document.querySelector("#chat-form"),
  chatInput: document.querySelector("#chat-input"),
  imageForm: document.querySelector("#image-form"),
  imageInput: document.querySelector("#image-input"),
  gallery: document.querySelector("#gallery"),
  toast: document.querySelector("#toast"),
};

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => {
    elements.toast.classList.remove("is-visible");
  }, 4200);
}

function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll("button[type='submit']").forEach((button) => {
    button.disabled = busy;
  });
}

function setMode(mode) {
  state.mode = mode;
  const isChat = mode === "chat";
  elements.chatView.classList.toggle("is-active", isChat);
  elements.imageView.classList.toggle("is-active", !isChat);
  elements.resetChat.hidden = !isChat;
  elements.viewEyebrow.textContent = isChat ? "CONVERSATION" : "IMAGE STUDIO";
  elements.viewTitle.textContent = isChat ? "和 Gemini 对话" : "把想法变成画面";
  elements.modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === mode);
  });
  window.requestAnimationFrame(() => {
    (isChat ? elements.chatInput : elements.imageInput).focus();
  });
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

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `请求失败（${response.status}）`);
  }
  return body;
}

async function loadStatus() {
  try {
    const status = await requestJson("/api/status");
    elements.statusDot.classList.add("is-ready");
    elements.statusText.textContent = `${status.account_status} · ${status.proxy_enabled ? "代理已启用" : "直连"}`;
    if (status.models?.length) {
      elements.modelSelect.replaceChildren(
        ...status.models.map((model) => {
          const option = document.createElement("option");
          option.value = model.name;
          option.textContent = model.label || model.name;
          return option;
        }),
      );
      const pro = status.models.find((model) => model.name.includes("pro"));
      if (pro) elements.modelSelect.value = pro.name;
    }
  } catch (error) {
    elements.statusDot.classList.add("is-error");
    elements.statusText.textContent = "连接失败";
    showToast(error.message);
  }
}

elements.modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

elements.resetChat.addEventListener("click", () => {
  state.sessionId = null;
  elements.conversation.replaceChildren();
  const intro = document.createElement("div");
  intro.className = "intro-copy";
  intro.id = "chat-intro";
  intro.innerHTML = '<span class="intro-star" aria-hidden="true">✦</span><h3>新对话已准备</h3><p>新的消息不会继承上一段对话的上下文。</p>';
  elements.conversation.append(intro);
  elements.chatInput.focus();
});

elements.modelSelect.addEventListener("change", () => {
  state.sessionId = null;
  showToast("模型已切换，下一条消息会开始新对话。");
});

elements.chatInput.addEventListener("input", () => {
  elements.chatInput.style.height = "auto";
  elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 160)}px`;
});

elements.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.chatForm.requestSubmit();
  }
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = elements.chatInput.value.trim();
  if (!message || state.busy) return;

  appendMessage("user", message);
  elements.chatInput.value = "";
  elements.chatInput.style.height = "auto";
  const pending = appendMessage("assistant", "正在思考");
  pending.classList.add("is-pending");
  setBusy(true);

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
    showToast(error.message);
  } finally {
    setBusy(false);
    elements.chatInput.focus();
  }
});

elements.imageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = elements.imageInput.value.trim();
  if (!prompt || state.busy) return;

  setBusy(true);
  elements.gallery.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "loading-copy";
  loading.textContent = "GEMINI 正在生成画面…";
  elements.gallery.append(loading);

  try {
    const result = await requestJson("/api/images", {
      method: "POST",
      body: JSON.stringify({ prompt, model: elements.modelSelect.value }),
    });
    const grid = document.createElement("div");
    grid.className = "gallery-grid";
    result.images.forEach((url, index) => {
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener";
      const image = document.createElement("img");
      image.src = url;
      image.alt = `生成图片 ${index + 1}：${prompt}`;
      link.append(image);
      grid.append(link);
    });
    elements.gallery.replaceChildren(grid);
  } catch (error) {
    elements.gallery.replaceChildren();
    showToast(error.message);
  } finally {
    setBusy(false);
  }
});

setMode("chat");
loadStatus();
