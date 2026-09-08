// Panther Lake AI Studio -- front end.
//
// Two views switched by hash routing: the home grid (#/) and one panel per
// available brick (#/brick/<id>). Leaving a panel never stops the brick --
// runs live on the server, the "Now running" strip under the header tracks
// them from /api/status, and reopening a panel rehydrates from that same
// status (plus a brick's own status route, where it has one).
//
// Every brick is a Panel (request/response: one call in, one result out) or a
// StreamPanel (a background run with Start/Stop -- results over a WebSocket,
// or an MJPEG video stream plus a polled side channel). The 13 configs at the
// bottom only carry what is genuinely different per brick.

const CATEGORY_ORDER = ["Speech", "Vision", "Text", "Productivity", "Audio"];

const el = (id) => document.getElementById(id);

// --- Small helpers ------------------------------------------------------------

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error || `${url} failed (${res.status})`);
  }
  return body;
}

function postJSON(url, body) {
  return fetchJSON(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

function option(value, label) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  return opt;
}

function fillSelect(select, items) {
  select.innerHTML = "";
  for (const item of items) select.appendChild(option(item.value, item.label));
}

function showPlaceholder(container, text) {
  container.innerHTML = `<p class="placeholder">${escapeHtml(text)}</p>`;
}

// Append one line to an output box, dropping its placeholder on first use and
// keeping the newest line in view.
function appendLine(container, className, content, { html = false } = {}) {
  const placeholder = container.querySelector(".placeholder");
  if (placeholder) placeholder.remove();
  const line = document.createElement("p");
  line.className = className;
  if (html) line.innerHTML = content;
  else line.textContent = content;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function appendTimedLine(container, time, lang, text) {
  appendLine(
    container,
    "line",
    `<span class="line-time">${escapeHtml(time)}</span><span class="line-lang">(${escapeHtml((lang || "auto").toUpperCase())})</span>${escapeHtml(text)}`,
    { html: true },
  );
}

function statRow(label, value) {
  return `<div class="stat-row"><span class="stat-label">${escapeHtml(label)}</span><span class="stat-value">${escapeHtml(value)}</span></div>`;
}

function renderTextBlock(container, text) {
  container.innerHTML = "";
  const block = document.createElement("p");
  block.className = "text-block";
  block.textContent = text;
  container.appendChild(block);
}

// --- Shared state ---------------------------------------------------------------

let DEMOS = [];
// Every OpenVINO-visible GPU on this machine ({id, full_name}[]), fetched once:
// drives the per-GPU gauges and the friendly labels in device dropdowns.
let GPU_DEVICES = [];
// Latest /api/status snapshot ({"<demo>[:<stage>]": {phase, message, at}})
// and the telemetry's active list ([{demo_id, device, stage_label}]).
const STATUS = { snapshot: {}, active: [] };

function demoById(id) {
  return DEMOS.find((d) => d.id === id);
}

function shortGpuName(fullName) {
  return fullName
    .replace(/Intel\(R\)\s*/g, "")
    .replace(/\(TM\)/g, "")
    .replace(/\s+Graphics/g, "")
    .replace(/\s+GPU/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function deviceLabel(id) {
  const upper = String(id).toUpperCase();
  if (upper.startsWith("GPU")) {
    const gpu = GPU_DEVICES.find((g) => g.id === id);
    return gpu ? gpu.full_name : id;
  }
  if (upper === "AUTO") return "AUTO (let OpenVINO choose)";
  if (upper === "CPU") return "CPU";
  if (upper === "NPU") return "NPU";
  if (upper === "CUDA") return "CUDA (NVIDIA GPU)";
  return id;
}

// The OpenVINO device to default a *large* model to (one that needs its own
// VRAM, e.g. a 30B coding LLM): the machine's discrete GPU if it has one and
// it's among the brick's offered devices, else AUTO. Mirrors
// pantherlake_ai_core.engine.preferred_large_model_device().
function preferredLargeModelDevice(openvinoDevices) {
  const discrete = GPU_DEVICES.filter(
    (g) => (g.full_name || "").includes("dGPU") && (openvinoDevices || []).includes(g.id),
  );
  return discrete.length ? discrete[discrete.length - 1].id : "AUTO";
}

// --- Control wiring ---------------------------------------------------------------

// One engine <select> + one compute-device <select>, kept consistent: the
// OpenVINO option is only enabled when this brick actually has a device, the
// device list follows the chosen engine (AUTO + every OpenVINO device, or the
// portable engine's fixed choices), and GPU ids get their friendly names.
function wireEngineAndDevice(engineSelect, deviceSelect, data, options = {}) {
  const { portableDevices = ["cpu"], preferLargeModel = false, onChange = null } = options;
  const openvinoDevices = data.openvino_devices || [];
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = openvinoDevices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  if (!hasOpenvino) openvinoOption.textContent = `${openvinoOption.textContent} -- not installed for this brick`;
  if (hasOpenvino) engineSelect.value = "openvino";

  const fill = () => {
    const isOpenvino = engineSelect.value === "openvino";
    const values = isOpenvino ? ["AUTO", ...openvinoDevices] : portableDevices;
    fillSelect(deviceSelect, values.map((value) => ({ value, label: deviceLabel(value) })));
    if (isOpenvino && preferLargeModel) deviceSelect.value = preferredLargeModelDevice(openvinoDevices);
    if (onChange) onChange(isOpenvino);
  };
  engineSelect.addEventListener("change", fill);
  fill();
}

// Microphone / output-device <select> that follows a mic-vs-system source pick.
function wireAudioSource(sourceSelect, deviceSelect, data) {
  const fill = () => {
    const names = sourceSelect.value === "mic" ? data.microphones || [] : data.speakers || [];
    fillSelect(deviceSelect, [{ value: "", label: "Default device" }, ...names.map((n) => ({ value: n, label: n }))]);
  };
  sourceSelect.addEventListener("change", fill);
  fill();
}

// Camera / screen <select> that follows a webcam-vs-screen source pick.
function wireVideoSource(sourceSelect, deviceSelect, data) {
  const fill = () => {
    if (sourceSelect.value === "webcam") {
      const cameras = data.cameras || [];
      fillSelect(
        deviceSelect,
        cameras.length
          ? cameras.map((index) => ({ value: String(index), label: `Camera ${index}` }))
          : [{ value: "", label: "No camera found" }],
      );
    } else if (sourceSelect.value === "screen") {
      fillSelect(
        deviceSelect,
        (data.screens || []).map((s) => ({ value: String(s.index), label: `Screen ${s.index} (${s.width}x${s.height})` })),
      );
    }
  };
  sourceSelect.addEventListener("change", fill);
  fill();
}

// "Try a sample" picker: fills one or more fields from the picked sample, then
// resets to its placeholder -- a one-shot insert, not a persistent choice.
// fieldMap is {targetElementId: sampleFieldName}; a null sample field leaves
// the target alone. A <select> target fires change (so show/hide logic runs),
// everything else fires input.
function wireSamplePicker(pickerId, samples, fieldMap) {
  const picker = el(pickerId);
  if (!picker) return;
  while (picker.options.length > 1) picker.remove(1);
  for (const sample of samples || []) {
    picker.appendChild(option(sample.name, `${sample.name} — ${sample.description}`));
  }
  picker.disabled = !(samples || []).length;
  picker.onchange = () => {
    const sample = (samples || []).find((s) => s.name === picker.value);
    if (sample) {
      for (const [targetId, sampleField] of Object.entries(fieldMap)) {
        const value = sample[sampleField];
        if (value === null || value === undefined) continue;
        const target = el(targetId);
        target.value = value;
        target.dispatchEvent(new Event(target.tagName === "SELECT" ? "change" : "input"));
      }
    }
    picker.value = "";
  };
}

// Recently used paths for a text input, kept per input in localStorage and
// offered through a <datalist> -- typing the same folder five times is the
// single most repetitive thing in these demos.
const RECENT_LIMIT = 6;

function recentPaths(inputId) {
  try {
    return JSON.parse(localStorage.getItem(`ptl.recent.${inputId}`) || "[]");
  } catch {
    return [];
  }
}

function refreshRecents(inputId) {
  const list = el(`${inputId}-recent`);
  if (!list) return;
  list.innerHTML = "";
  for (const value of recentPaths(inputId)) list.appendChild(option(value, value));
}

function attachRecents(inputId) {
  const input = el(inputId);
  if (!input || el(`${inputId}-recent`)) return;
  const list = document.createElement("datalist");
  list.id = `${inputId}-recent`;
  input.insertAdjacentElement("afterend", list);
  input.setAttribute("list", list.id);
  refreshRecents(inputId);
}

function rememberPath(inputId) {
  const value = el(inputId).value.trim();
  if (!value) return;
  const next = [value, ...recentPaths(inputId).filter((v) => v !== value)].slice(0, RECENT_LIMIT);
  try {
    localStorage.setItem(`ptl.recent.${inputId}`, JSON.stringify(next));
  } catch {
    // Storage unavailable (private mode, quota) -- the convenience just doesn't persist.
  }
  refreshRecents(inputId);
}

// --- Status pills ------------------------------------------------------------------

const PHASE_ICON = { loading: "", running: "▶", stopping: "", error: "⚠" };
// "stopping" borrows the loading look -- a pulsing amber dot -- because it
// is the same kind of state: something is in flight and the user waits.
const PHASE_KIND = { loading: "loading", running: "live", stopping: "loading", error: "error" };
const PHASE_LABEL = { loading: "Loading…", running: "Running", stopping: "Stopping…" };

function paintStatus(target, text, kind) {
  target.textContent = text;
  target.classList.remove("live", "error", "loading");
  if (kind) target.classList.add(kind);
}

function reflectStatus(target, status) {
  const icon = PHASE_ICON[status.phase] || "";
  paintStatus(target, `${icon} ${status.message || status.phase}`.trim(), PHASE_KIND[status.phase] || "loading");
}

// --- Panels -------------------------------------------------------------------

class Panel {
  constructor(config) {
    Object.assign(this, { controls: [], portableDevices: ["cpu"] }, config);
    this.populated = false;
    this.isOpen = false;
    this.watchKey = null;
    this.watchTarget = null;
    this.timers = [];
  }

  get statusEl() {
    return el(`${this.prefix}-status`);
  }

  setStatus(text, kind) {
    paintStatus(this.statusEl, text, kind);
  }

  hasError() {
    return this.statusEl.classList.contains("error");
  }

  // Mirror the brick's real backend phase (from the shared /api/status poll)
  // into a pill until unwatch() -- so a slow first-time download or compile
  // shows what is actually happening instead of a static "Working...".
  watch(key, target = this.statusEl) {
    this.watchKey = key;
    this.watchTarget = target;
    this.onStatus();
  }

  unwatch() {
    this.watchKey = null;
    this.watchTarget = null;
  }

  onStatus() {
    if (!this.watchKey || !this.isOpen) return;
    const status = STATUS.snapshot[this.watchKey];
    if (status) reflectStatus(this.watchTarget, status);
  }

  every(ms, fn) {
    this.timers.push(setInterval(fn, ms));
  }

  clearTimers() {
    for (const timer of this.timers) clearInterval(timer);
    this.timers = [];
  }

  async open() {
    this.isOpen = true;
    if (!this.populated) {
      try {
        const data = await fetchJSON(`/api/${this.id}/devices`);
        this.populate(data);
        this.populated = true;
      } catch (err) {
        this.setStatus(`Error: ${err.message}`, "error");
        return;
      }
    }
    await this.rehydrate();
  }

  leave() {
    this.isOpen = false;
    this.unwatch();
    this.clearTimers();
  }

  populate() {}

  async rehydrate() {}

  wire() {}

  // One request/response action: disables `button`, shows `busy`, mirrors the
  // brick's phase (statusKey) into `statusEl` while the call is in flight,
  // then `done` (a string, or a function of the result) -- or the error.
  async run({ button, statusEl = this.statusEl, key, busy, work, done = "Done" }) {
    button.disabled = true;
    paintStatus(statusEl, busy, "loading");
    if (key) this.watch(key, statusEl);
    try {
      const result = await work();
      this.unwatch();
      paintStatus(statusEl, typeof done === "function" ? done(result) : done, "live");
      return result;
    } catch (err) {
      this.unwatch();
      paintStatus(statusEl, `Error: ${err.message}`, "error");
      return undefined;
    } finally {
      button.disabled = false;
    }
  }
}

class StreamPanel extends Panel {
  constructor(config) {
    super(config);
    this.running = false;
    this.ws = null;
  }

  get img() {
    return this.video ? el(this.video) : null;
  }

  async open() {
    await super.open();
    if (this.transport === "ws" && this.isOpen) this.connect();
  }

  leave() {
    super.leave();
    this.disconnect();
    if (this.img) this.detachVideo();
  }

  // The brick may have been started from a previous visit (or before a page
  // reload): pick up its real state instead of assuming Idle.
  async rehydrate() {
    const status = STATUS.snapshot[this.statusKey];
    if (status && status.phase !== "error") {
      this.setRunning(true);
    } else {
      this.setRunning(false);
      if (status) reflectStatus(this.statusEl, status);
    }
  }

  setRunning(isRunning) {
    this.running = isRunning;
    el(`${this.prefix}-start`).disabled = isRunning;
    el(`${this.prefix}-stop`).disabled = !isRunning;
    for (const id of this.controls) el(id).disabled = isRunning;
    if (isRunning) {
      // Whether this run has ever shown up in /api/status yet. Until it
      // has, a missing entry means "the thread hasn't reported in", not
      // "it finished" -- see onStatus().
      this.sawPhase = Boolean(STATUS.snapshot[this.statusKey]);
      if (!this.sawPhase) this.setStatus("Starting…", "loading");
      this.watch(this.statusKey);
      if (this.img) this.attachVideo();
      if (this.onRunning) this.onRunning(true);
    } else {
      this.unwatch();
      this.clearTimers();
      if (!this.hasError()) this.setStatus("Idle");
      if (this.img) this.detachVideo();
      if (this.onRunning) this.onRunning(false);
    }
  }

  // A run can end without this panel asking it to: a batch finishing, a
  // video reaching its end, or a stop the brick was too busy to honour
  // right away. The phase vanishing from /api/status is what says so.
  onStatus() {
    super.onStatus();
    if (!this.isOpen || !this.running) return;
    if (STATUS.snapshot[this.statusKey]) this.sawPhase = true;
    else if (this.sawPhase) this.setRunning(false);
  }

  streamUrl() {
    return `/api/${this.id}/stream?t=${Date.now()}`;
  }

  attachVideo() {
    const url = this.streamUrl();
    if (!url) return;
    this.img.src = url;
    this.img.classList.add("visible");
  }

  detachVideo() {
    this.img.removeAttribute("src");
    this.img.classList.remove("visible");
  }

  async start() {
    let body;
    try {
      body = this.body();
    } catch (err) {
      this.setStatus(err.message, "error");
      return;
    }
    this.setStatus("Starting…", "loading");
    try {
      const data = await postJSON(`/api/${this.id}/start`, body);
      if (this.onStarted) this.onStarted(data);
      this.setRunning(true);
    } catch (err) {
      this.setStatus(`Error: ${err.message}`, "error");
    }
  }

  async stop() {
    el(`${this.prefix}-stop`).disabled = true;
    this.setStatus("Stopping…", "loading");
    try {
      await postJSON(`/api/${this.id}/stop`);
    } catch (err) {
      this.setStatus(`Error: ${err.message}`, "error");
      this.setRunning(false);
      return;
    }
    // A stop event is cooperative: a brick inside a model load or one long
    // inference keeps going until that returns. When it does, say so and
    // leave the run's controls locked -- it really is still running --
    // rather than showing Idle over a brick that is still working.
    // onStatus() flips to idle once the phase clears.
    await pollStatus();
    if (STATUS.snapshot[this.statusKey]?.phase === "stopping") {
      this.watch(this.statusKey);
      return;
    }
    this.setRunning(false);
  }

  connect() {
    if (this.ws) return;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws/${this.id}`);
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "error") {
        this.setStatus(`Error: ${message.message}`, "error");
        this.setRunning(false);
      } else if (message.type === "stopped") {
        this.setRunning(false);
        if (this.onStopped) this.onStopped();
      } else if (this.onMessage) {
        this.onMessage(message);
      }
    };
    // The run is a server-side thread that outlives any one socket: while the
    // panel is open, a dropped connection just reconnects.
    ws.onclose = () => {
      this.ws = null;
      if (this.isOpen) setTimeout(() => this.connect(), 1000);
    };
    this.ws = ws;
  }

  disconnect() {
    if (!this.ws) return;
    this.ws.onclose = null;
    this.ws.close();
    this.ws = null;
  }

  wire() {
    el(`${this.prefix}-start`).addEventListener("click", () => this.start());
    el(`${this.prefix}-stop`).addEventListener("click", () => this.stop());
    if (this.wireExtra) this.wireExtra();
  }
}

// --- Brick configs --------------------------------------------------------------------

const PANELS = {
  "live-translation": new StreamPanel({
    id: "live-translation",
    prefix: "lt",
    transport: "ws",
    statusKey: "live-translation",
    controls: ["lt-source", "lt-audio-device", "lt-engine", "lt-compute-device", "lt-model"],
    populate(data) {
      wireAudioSource(el("lt-source"), el("lt-audio-device"), data);
      const modelSelect = el("lt-model");
      const small = modelSelect.querySelector('option[value="small"]');
      wireEngineAndDevice(el("lt-engine"), el("lt-compute-device"), data, {
        portableDevices: ["cpu", "cuda"],
        onChange: (isOpenvino) => {
          // Intel publishes pre-converted OpenVINO Whisper for
          // tiny/base/medium/large-v3 only -- there is no "small".
          small.disabled = isOpenvino;
          small.textContent = isOpenvino ? "small (portable engine only)" : "small";
          modelSelect.value = isOpenvino ? "base" : "small";
        },
      });
    },
    body() {
      return {
        source: el("lt-source").value,
        audio_device: el("lt-audio-device").value || null,
        engine: el("lt-engine").value,
        model_size: el("lt-model").value,
        compute_device: el("lt-compute-device").value,
      };
    },
    onMessage(message) {
      if (message.type === "result") {
        appendTimedLine(el("lt-transcript"), new Date().toLocaleTimeString(), message.detected_language, message.text);
      }
    },
  }),

  "meeting-notes": new StreamPanel({
    id: "meeting-notes",
    prefix: "mtg",
    transport: "ws",
    statusKey: "meeting-notes",
    controls: ["mtg-source", "mtg-audio-device", "mtg-engine", "mtg-compute-device"],
    populate(data) {
      wireAudioSource(el("mtg-source"), el("mtg-audio-device"), data);
      wireEngineAndDevice(el("mtg-engine"), el("mtg-compute-device"), data, { portableDevices: ["cpu", "cuda"] });
    },
    body() {
      return {
        source: el("mtg-source").value,
        audio_device: el("mtg-audio-device").value || null,
        engine: el("mtg-engine").value,
        compute_device: el("mtg-compute-device").value,
      };
    },
    onMessage(message) {
      if (message.type === "line") {
        appendTimedLine(el("mtg-transcript"), message.timestamp, message.detected_language, message.text);
      }
    },
    wireExtra() {
      el("mtg-generate").addEventListener("click", () =>
        this.run({
          button: el("mtg-generate"),
          statusEl: el("mtg-notes-status"),
          key: "meeting-notes:notes",
          busy: "Generating notes…",
          work: async () => {
            const data = await postJSON("/api/meeting-notes/generate");
            renderTextBlock(el("mtg-notes"), data.text);
            return data;
          },
          done: (data) => `Based on ${data.transcript_line_count} transcript line(s)`,
        }),
      );
    },
  }),

  "voice-assistant": new StreamPanel({
    id: "voice-assistant",
    prefix: "va",
    transport: "ws",
    statusKey: "voice-assistant",
    controls: ["va-audio-device", "va-wake-word", "va-engine", "va-compute-device", "va-speak"],
    populate(data) {
      fillSelect(el("va-audio-device"), [
        { value: "", label: "Default microphone" },
        ...(data.microphones || []).map((n) => ({ value: n, label: n })),
      ]);
      fillSelect(el("va-wake-word"), (data.wake_words || []).map((w) => ({ value: w, label: w.replace(/_/g, " ") })));
      wireEngineAndDevice(el("va-engine"), el("va-compute-device"), data);
    },
    body() {
      return {
        audio_device: el("va-audio-device").value || null,
        engine: el("va-engine").value,
        compute_device: el("va-compute-device").value,
        wake_word: el("va-wake-word").value,
        speak_replies: el("va-speak").checked,
      };
    },
    onMessage(message) {
      const box = el("va-transcript");
      if (message.type === "wake") appendLine(box, "line-note", "Wake word heard -- listening for your question...");
      else if (message.type === "heard") appendLine(box, "line-question", `You: ${message.text}`);
      else if (message.type === "reply") appendLine(box, "line-answer", `Assistant: ${message.text}`);
    },
  }),

  "expense-extract": new StreamPanel({
    id: "expense-extract",
    prefix: "expx",
    transport: "ws",
    statusKey: "expense-extract:ocr",
    controls: ["expx-folder", "expx-sample", "expx-ocr-engine", "expx-ocr-device", "expx-llm-engine", "expx-llm-device"],
    populate(data) {
      attachRecents("expx-folder");
      wireEngineAndDevice(el("expx-ocr-engine"), el("expx-ocr-device"), data);
      wireEngineAndDevice(el("expx-llm-engine"), el("expx-llm-device"), data);
      // Nudge the demo toward its actual point: OCR on a GPU, LLM structuring
      // on the NPU, at once -- if this machine has both. GPU for OCR, not the
      // other way around: screen-ocr's OpenVINO engine is a 7B vision-language
      // model whose NPU compile fails on this hardware (see expense-extract's
      // README); doc-qa's small LLM compiles and runs fine on the NPU.
      const devices = data.openvino_devices || [];
      const anyGpu = devices.find((d) => d.toUpperCase().startsWith("GPU"));
      if (anyGpu) el("expx-ocr-device").value = anyGpu;
      if (devices.includes("NPU")) el("expx-llm-device").value = "NPU";
      wireSamplePicker("expx-sample", data.samples, { "expx-folder": "folder" });
    },
    body() {
      const folder = el("expx-folder").value.trim();
      if (!folder) throw new Error("Enter a folder of receipt photos first.");
      rememberPath("expx-folder");
      return {
        folder,
        ocr_engine: el("expx-ocr-engine").value,
        ocr_compute_device: el("expx-ocr-device").value,
        llm_engine: el("expx-llm-engine").value,
        llm_compute_device: el("expx-llm-device").value,
      };
    },
    onStarted() {
      showPlaceholder(el("expx-transcript"), "Each receipt's vendor, date, amount, and category will appear here as it is structured.");
    },
    onMessage(message) {
      const box = el("expx-transcript");
      if (message.type === "ocr_progress") {
        appendLine(box, "line-note", `Reading receipt ${message.index}/${message.total}: ${message.file}`);
      } else if (message.type === "structured") {
        const line = message.line;
        if (line.error) {
          appendLine(box, "line-answer", `${line.source_file}: skipped (${line.error})`);
        } else {
          const amount = line.amount !== null && line.amount !== undefined ? `$${line.amount.toFixed(2)}` : "?";
          appendLine(box, "line-answer", `${line.source_file}: ${line.vendor || "?"} -- ${line.date || "?"} -- ${amount} -- ${line.category}`);
        }
      } else if (message.type === "done") {
        this.setRunning(false);
        this.setStatus(`Done -- ${message.structured}/${message.count} structured, total $${message.total.toFixed(2)}`, "live");
      }
    },
  }),

  "smart-recall": new StreamPanel({
    id: "smart-recall",
    prefix: "recall",
    transport: "ws",
    statusKey: "smart-recall:ocr",
    // The embedding engine is locked/unlocked by the index's own state (see
    // refreshStatus), not by whether recording is running.
    controls: ["recall-screen", "recall-interval", "recall-ocr-engine", "recall-ocr-device", "recall-embed-device"],
    populate(data) {
      fillSelect(el("recall-screen"), (data.screens || []).map((s) => ({ value: String(s.index), label: `Screen ${s.index} (${s.width}x${s.height})` })));
      wireEngineAndDevice(el("recall-ocr-engine"), el("recall-ocr-device"), data);
      wireEngineAndDevice(el("recall-embed-engine"), el("recall-embed-device"), data);
      wireSamplePicker("recall-sample", data.samples, { "recall-question": "question" });
    },
    async rehydrate() {
      await this.refreshStatus();
    },
    async refreshStatus() {
      const status = await fetchJSON("/api/smart-recall/status");
      this.setRunning(status.running);
      const embedEngine = el("recall-embed-engine");
      if (status.embed_engine) {
        // The index already has a fixed embedding engine -- lock the pick to
        // it rather than offer something that would just be rejected.
        embedEngine.value = status.embed_engine;
        embedEngine.dispatchEvent(new Event("change"));
        embedEngine.disabled = true;
      } else {
        embedEngine.disabled = status.running;
      }
      if (!status.running && !this.hasError()) {
        this.setStatus(status.indexed_count > 0 ? `Idle -- ${status.indexed_count} screen(s) indexed` : "Idle");
      }
    },
    body() {
      return {
        screen_index: Number(el("recall-screen").value),
        interval_seconds: Number(el("recall-interval").value) || 5,
        ocr_engine: el("recall-ocr-engine").value,
        ocr_compute_device: el("recall-ocr-device").value,
        embed_engine: el("recall-embed-engine").value,
        embed_compute_device: el("recall-embed-device").value,
      };
    },
    onRunning(isRunning) {
      el("recall-reset").disabled = isRunning;
    },
    onMessage(message) {
      const box = el("recall-capture-feed");
      if (message.type === "indexed") appendLine(box, "line-answer", `[${message.timestamp}] indexed: ${message.chunk.text.slice(0, 100)}`);
      else if (message.type === "skipped") appendLine(box, "line-note", `(skipped -- ${message.reason})`);
    },
    onStopped() {
      this.refreshStatus().catch(() => {});
    },
    wireExtra() {
      el("recall-reset").addEventListener("click", async () => {
        if (!confirm("Delete every indexed screen capture and screenshot? This can't be undone.")) return;
        try {
          await postJSON("/api/smart-recall/reset");
          el("recall-embed-engine").disabled = false;
          showPlaceholder(el("recall-capture-feed"), "Capture events will appear here while recording.");
          showPlaceholder(el("recall-results"), "Search results, each with a screenshot thumbnail, will appear here.");
          await this.refreshStatus();
        } catch (err) {
          this.setStatus(`Error: ${err.message}`, "error");
        }
      });
      const search = async () => {
        const question = el("recall-question").value.trim();
        if (!question) return;
        el("recall-search").disabled = true;
        try {
          const data = await postJSON("/api/smart-recall/search", { question, top_k: 5 });
          this.renderResults(data.results || []);
        } catch (err) {
          showPlaceholder(el("recall-results"), `Error: ${err.message}`);
        } finally {
          el("recall-search").disabled = false;
        }
      };
      el("recall-search").addEventListener("click", search);
      el("recall-question").addEventListener("keydown", (event) => {
        if (event.key === "Enter") search();
      });
    },
    renderResults(results) {
      const container = el("recall-results");
      if (!results.length) {
        showPlaceholder(container, "No matches yet.");
        return;
      }
      container.innerHTML = results
        .map(
          (r) => `
        <div class="recall-result">
          <img class="recall-result-thumb" src="${escapeHtml(r.screenshot_url)}" alt="Screenshot from ${escapeHtml(r.source)}" />
          <div class="recall-result-body">
            <div class="recall-result-meta"><span>${escapeHtml(r.source)}</span><span class="recall-result-score">${r.score.toFixed(2)}</span></div>
            <p class="recall-result-text">${escapeHtml(r.text.slice(0, 220))}</p>
          </div>
        </div>`,
        )
        .join("");
    },
  }),

  "object-detection": new StreamPanel({
    id: "object-detection",
    prefix: "objdet",
    transport: "mjpeg",
    video: "objdet-video",
    statusKey: "object-detection",
    controls: ["objdet-source", "objdet-source-device", "objdet-engine", "objdet-compute-device"],
    populate(data) {
      wireVideoSource(el("objdet-source"), el("objdet-source-device"), data);
      wireEngineAndDevice(el("objdet-engine"), el("objdet-compute-device"), data);
    },
    body() {
      const source = el("objdet-source").value;
      const device = el("objdet-source-device").value;
      return {
        source,
        camera_index: source === "webcam" ? Number(device || 0) : 0,
        screen_index: source === "screen" ? Number(device || 1) : 1,
        engine: el("objdet-engine").value,
        compute_device: el("objdet-compute-device").value,
      };
    },
    onRunning(isRunning) {
      const box = el("objdet-detections");
      if (!isRunning) {
        showPlaceholder(box, "Detected objects will be listed here.");
        return;
      }
      this.every(700, async () => {
        try {
          const data = await fetchJSON("/api/object-detection/detections");
          if (data.error) {
            this.setStatus(`Error: ${data.error}`, "error");
            this.setRunning(false);
            return;
          }
          const detections = [...(data.detections || [])].sort((a, b) => b.confidence - a.confidence);
          box.innerHTML = detections.length
            ? detections.map((d) => statRow(d.label, `${Math.round(d.confidence * 100)}%`)).join("")
            : '<p class="placeholder">Nothing detected right now.</p>';
        } catch {
          // Best-effort -- a transient failure shouldn't interrupt the video stream.
        }
      });
    },
  }),

  "webcam-effects": new StreamPanel({
    id: "webcam-effects",
    prefix: "webcam",
    transport: "mjpeg",
    video: "webcam-video",
    statusKey: "webcam-effects",
    controls: ["webcam-camera", "webcam-engine", "webcam-compute-device"],
    populate(data) {
      const cameras = data.cameras || [];
      fillSelect(
        el("webcam-camera"),
        cameras.length
          ? cameras.map((index) => ({ value: String(index), label: `Camera ${index}` }))
          : [{ value: "", label: "No camera found" }],
      );
      wireEngineAndDevice(el("webcam-engine"), el("webcam-compute-device"), data);
      const effect = el("webcam-effect");
      const colorField = el("webcam-color-field");
      const sync = () => {
        colorField.hidden = effect.value !== "replace";
        if (this.running) this.sendEffect();
      };
      effect.addEventListener("change", sync);
      el("webcam-color").addEventListener("change", () => {
        if (this.running) this.sendEffect();
      });
      sync();
    },
    async sendEffect() {
      try {
        await postJSON("/api/webcam-effects/effect", { effect: el("webcam-effect").value, color: el("webcam-color").value });
      } catch {
        // Best-effort -- a failed live switch just leaves the previous look on screen.
      }
    },
    body() {
      return {
        camera_index: Number(el("webcam-camera").value || 0),
        engine: el("webcam-engine").value,
        compute_device: el("webcam-compute-device").value,
        effect: el("webcam-effect").value,
        color: el("webcam-color").value,
      };
    },
    onRunning(isRunning) {
      const box = el("webcam-stats");
      if (!isRunning) {
        showPlaceholder(box, "How much of the frame the model sees as a person will be shown here.");
        return;
      }
      this.every(700, async () => {
        try {
          const data = await fetchJSON("/api/webcam-effects/stats");
          if (data.error) {
            this.setStatus(`Error: ${data.error}`, "error");
            this.setRunning(false);
            return;
          }
          box.innerHTML = statRow("Person coverage", `${Math.round(data.person_coverage * 100)}%`);
        } catch {
          // Best-effort.
        }
      });
    },
  }),

  "smart-city-monitor": new StreamPanel({
    id: "smart-city-monitor",
    prefix: "smartcity",
    transport: "mjpeg",
    video: "smartcity-video",
    statusKey: "smart-city-monitor:feed-1",
    controls: ["smartcity-feeds", "smartcity-engine", "smartcity-compute-device", "smartcity-loop"],
    feeds: [],
    populate(data) {
      wireEngineAndDevice(el("smartcity-engine"), el("smartcity-compute-device"), data);
      el("smartcity-feed-picker").addEventListener("change", () => this.attachVideo());
    },
    async rehydrate() {
      // The feed list lives on the server (a run may predate this visit):
      // pick it up from the counts route, then the usual running check.
      try {
        const data = await fetchJSON("/api/smart-city-monitor/counts");
        this.feeds = data.feeds || [];
      } catch {
        this.feeds = [];
      }
      await StreamPanel.prototype.rehydrate.call(this);
    },
    // "path" or "path|device" per line; a blank device means the shared pick.
    body() {
      const feeds = el("smartcity-feeds")
        .value.split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const pipe = line.lastIndexOf("|");
          if (pipe === -1) return { path: line, compute_device: null };
          return { path: line.slice(0, pipe), compute_device: line.slice(pipe + 1).trim() || null };
        });
      if (!feeds.length) throw new Error("Enter at least one video file path.");
      return {
        feeds,
        engine: el("smartcity-engine").value,
        compute_device: el("smartcity-compute-device").value,
        loop: el("smartcity-loop").checked,
      };
    },
    onStarted(data) {
      this.feeds = data.feeds || [];
    },
    streamUrl() {
      const feedId = el("smartcity-feed-picker").value;
      return feedId ? `/api/smart-city-monitor/stream?feed=${encodeURIComponent(feedId)}&t=${Date.now()}` : null;
    },
    onRunning(isRunning) {
      const picker = el("smartcity-feed-picker");
      const combined = el("smartcity-combined-counts");
      const perFeed = el("smartcity-per-feed");
      if (!isRunning) {
        el("smartcity-picker-row").hidden = true;
        picker.innerHTML = "";
        showPlaceholder(combined, "Combined per-minute counts will appear here once running.");
        perFeed.innerHTML = "";
        this.feeds = [];
        return;
      }
      fillSelect(picker, this.feeds.map((f) => ({ value: f.feed_id, label: `${f.name} -- ${f.compute_device}` })));
      el("smartcity-picker-row").hidden = this.feeds.length === 0;
      this.attachVideo();
      this.every(1000, async () => {
        try {
          const data = await fetchJSON("/api/smart-city-monitor/counts");
          if (data.error) {
            this.setStatus(`Error: ${data.error}`, "error");
            this.setRunning(false);
            return;
          }
          if (!data.snapshot) return;
          const entries = Object.entries(data.snapshot.combined_last_60s || {});
          combined.innerHTML = entries.length
            ? entries.map(([label, count]) => statRow(label, `${count}/min`)).join("")
            : '<p class="placeholder">No relevant objects counted yet.</p>';
          perFeed.innerHTML = this.feeds
            .map((feed) => {
              const counts = (data.snapshot.per_feed_last_60s || {})[feed.feed_id] || {};
              const parts = Object.entries(counts).map(([label, count]) => `${escapeHtml(label)}: ${count}/min`).join(", ") || "nothing counted yet";
              return `<p class="feed-summary"><strong>${escapeHtml(feed.name)}</strong> (${escapeHtml(feed.compute_device)}): ${parts}</p>`;
            })
            .join("");
        } catch {
          // Best-effort.
        }
      });
    },
  }),

  "doc-qa": new Panel({
    id: "doc-qa",
    prefix: "docqa",
    indexed: false,
    populate(data) {
      attachRecents("docqa-folder");
      wireEngineAndDevice(el("docqa-engine"), el("docqa-compute-device"), data);
      wireSamplePicker("docqa-sample", data.samples, { "docqa-folder": "folder", "docqa-question": "question" });
    },
    async rehydrate() {
      // An index built on a previous visit (or before a reload) is still
      // loaded server-side -- reflect it instead of asking to re-index.
      try {
        const status = await fetchJSON("/api/doc-qa/status");
        this.setIndexed(status.indexed);
        if (status.indexed) {
          this.setStatus(`Indexed ${status.chunks} chunk(s) from ${status.folder}`, "live");
          if (!el("docqa-folder").value) el("docqa-folder").value = status.folder;
        }
      } catch {
        this.setIndexed(false);
      }
    },
    setIndexed(indexed) {
      this.indexed = indexed;
      this.setBusy(false);
    },
    setBusy(busy) {
      el("docqa-ingest").disabled = busy;
      el("docqa-ask").disabled = busy || !this.indexed;
      el("docqa-question").disabled = busy || !this.indexed;
      // The sample picker also fills the folder -- usable before indexing, so
      // only gated on busy.
      for (const id of ["docqa-folder", "docqa-engine", "docqa-compute-device", "docqa-reindex", "docqa-sample"]) {
        el(id).disabled = busy;
      }
    },
    wire() {
      el("docqa-ingest").addEventListener("click", async () => {
        const folder = el("docqa-folder").value.trim();
        if (!folder) {
          this.setStatus("Enter a folder path first.", "error");
          return;
        }
        this.indexed = false;
        this.setBusy(true);
        const result = await this.run({
          button: el("docqa-ingest"),
          key: "doc-qa",
          busy: "Indexing… (the first run downloads the models)",
          work: () =>
            postJSON("/api/doc-qa/ingest", {
              folder,
              engine: el("docqa-engine").value,
              compute_device: el("docqa-compute-device").value,
              reindex: el("docqa-reindex").checked,
            }),
          done: (r) => `Indexed ${r.chunks} chunk(s) from ${r.folder}`,
        });
        if (result) rememberPath("docqa-folder");
        this.setIndexed(Boolean(result));
      });
      const ask = async () => {
        const question = el("docqa-question").value.trim();
        if (!question) return;
        const box = el("docqa-transcript");
        appendLine(box, "line-question", `Q: ${question}`);
        el("docqa-question").value = "";
        this.setBusy(true);
        try {
          const answer = await postJSON("/api/doc-qa/ask", { question });
          appendLine(box, "line-answer", answer.text);
          if (answer.sources && answer.sources.length) {
            appendLine(box, "line-note", "Sources: " + answer.sources.map((s) => `${s.source} [${s.score.toFixed(2)}]`).join(", "));
          }
        } catch (err) {
          appendLine(box, "line-answer", `Error: ${err.message}`);
        } finally {
          this.setBusy(false);
        }
      };
      el("docqa-ask").addEventListener("click", ask);
      el("docqa-question").addEventListener("keydown", (event) => {
        if (event.key === "Enter") ask();
      });
    },
  }),

  "screen-ocr": new Panel({
    id: "screen-ocr",
    prefix: "ocr",
    populate(data) {
      const source = el("ocr-source");
      const sync = () => {
        const isUpload = source.value === "upload";
        el("ocr-source-device-field").hidden = isUpload;
        el("ocr-upload-field").hidden = !isUpload;
      };
      wireVideoSource(source, el("ocr-source-device"), data);
      source.addEventListener("change", sync);
      sync();
      const translate = el("ocr-translate");
      wireEngineAndDevice(el("ocr-engine"), el("ocr-compute-device"), data, {
        onChange: (isOpenvino) => {
          translate.disabled = !isOpenvino;
          if (!isOpenvino) translate.checked = false;
        },
      });
    },
    wire() {
      el("ocr-extract").addEventListener("click", () => {
        const source = el("ocr-source").value;
        const engine = el("ocr-engine").value;
        const computeDevice = el("ocr-compute-device").value;
        const translate = el("ocr-translate").checked;
        if (source === "upload" && !el("ocr-upload").files.length) {
          this.setStatus("Choose an image file first.", "error");
          return;
        }
        this.run({
          button: el("ocr-extract"),
          key: "screen-ocr",
          busy: source === "upload" ? "Uploading and reading…" : "Capturing and reading…",
          work: async () => {
            let data;
            if (source === "upload") {
              const form = new FormData();
              form.append("file", el("ocr-upload").files[0]);
              form.append("engine", engine);
              form.append("compute_device", computeDevice);
              form.append("translate", translate);
              data = await fetchJSON("/api/screen-ocr/extract-upload", { method: "POST", body: form });
            } else {
              const device = el("ocr-source-device").value;
              data = await postJSON("/api/screen-ocr/extract", {
                source,
                camera_index: source === "webcam" ? Number(device || 0) : 0,
                screen_index: source === "screen" ? Number(device || 1) : 1,
                engine,
                compute_device: computeDevice,
                translate,
              });
            }
            this.renderResult(data);
            return data;
          },
        });
      });
    },
    renderResult(data) {
      const container = el("ocr-result");
      const translated = data.translated_text !== null && data.translated_text !== undefined;
      const shown = translated ? data.translated_text : data.text;
      let html = `<p class="section-label">${translated ? "Translation (English)" : "Extracted text"}</p>`;
      html += `<p class="text-block">${escapeHtml(shown || "(no text detected)")}</p>`;
      if (data.regions && data.regions.length) {
        html += '<p class="section-label">Detected regions</p>';
        html += data.regions.map((r) => statRow(r.text, `${Math.round(r.confidence * 100)}%`)).join("");
      }
      container.innerHTML = html;
    },
  }),

  "voice-clone-studio": new Panel({
    id: "voice-clone-studio",
    prefix: "voice",
    populate(data) {
      const source = el("voice-source");
      const sync = () => {
        const isUpload = source.value === "upload";
        el("voice-record-field").hidden = isUpload;
        el("voice-upload-field").hidden = !isUpload;
      };
      source.addEventListener("change", sync);
      if (!(data.microphones || []).length) {
        const record = source.querySelector('option[value="record"]');
        record.disabled = true;
        record.textContent = "Record from the microphone (none found)";
        source.value = "upload";
      }
      sync();
      wireEngineAndDevice(el("voice-engine"), el("voice-compute-device"), data);
      const tau = el("voice-tau");
      const readout = el("voice-tau-value");
      tau.addEventListener("input", () => {
        readout.value = Number(tau.value).toFixed(2);
      });
      wireSamplePicker("voice-sample", data.samples, { "voice-text": "text" });
    },
    async rehydrate() {
      try {
        const status = await fetchJSON("/api/voice-clone-studio/status");
        this.setEnrolled(status.enrolled);
      } catch {
        this.setEnrolled(false);
      }
    },
    setEnrolled(enrolled) {
      for (const id of ["voice-text", "voice-sample", "voice-style", "voice-tau", "voice-synthesize"]) {
        el(id).disabled = !enrolled;
      }
      if (enrolled) paintStatus(el("voice-enroll-status"), "Voice enrolled -- ready to speak", "live");
    },
    wire() {
      el("voice-enroll").addEventListener("click", () => {
        const source = el("voice-source").value;
        const engine = el("voice-engine").value;
        const computeDevice = el("voice-compute-device").value;
        if (source === "upload" && !el("voice-upload").files.length) {
          paintStatus(el("voice-enroll-status"), "Choose an audio file first.", "error");
          return;
        }
        this.run({
          button: el("voice-enroll"),
          statusEl: el("voice-enroll-status"),
          key: "voice-clone-studio",
          busy: source === "record" ? "Recording…" : "Uploading and enrolling…",
          work: async () => {
            if (source === "upload") {
              const form = new FormData();
              form.append("file", el("voice-upload").files[0]);
              form.append("engine", engine);
              form.append("compute_device", computeDevice);
              await fetchJSON("/api/voice-clone-studio/enroll-upload", { method: "POST", body: form });
            } else {
              await postJSON("/api/voice-clone-studio/enroll-record", {
                seconds: Number(el("voice-record-seconds").value) || 10,
                engine,
                compute_device: computeDevice,
              });
            }
            this.setEnrolled(true);
            return true;
          },
          done: "Voice enrolled -- ready to speak",
        });
      });
      el("voice-synthesize").addEventListener("click", () => {
        const text = el("voice-text").value.trim();
        if (!text) {
          this.setStatus("Type something to say first.", "error");
          return;
        }
        this.run({
          button: el("voice-synthesize"),
          key: "voice-clone-studio",
          busy: "Synthesizing…",
          work: async () => {
            const res = await fetch("/api/voice-clone-studio/synthesize", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text, style: el("voice-style").value, tau: Number(el("voice-tau").value) }),
            });
            if (!res.ok) {
              const body = await res.json().catch(() => ({}));
              throw new Error(body.error || `synthesize failed (${res.status})`);
            }
            const player = el("voice-player");
            player.src = URL.createObjectURL(await res.blob());
            player.hidden = false;
            player.play().catch(() => {});
            return true;
          },
        });
      });
    },
  }),

  "code-review-assist": new Panel({
    id: "code-review-assist",
    prefix: "cra",
    populate(data) {
      attachRecents("cra-folder");
      const source = el("cra-source");
      const sync = () => {
        const isWorktree = source.value === "worktree";
        el("cra-folder-field").hidden = !isWorktree;
        el("cra-against-field").hidden = !isWorktree;
        el("cra-diff-text-field").hidden = isWorktree;
      };
      source.addEventListener("change", sync);
      sync();
      wireEngineAndDevice(el("cra-engine"), el("cra-compute-device"), data, { preferLargeModel: true });
      wireSamplePicker("cra-sample", data.samples, { "cra-source": "source", "cra-diff-text": "diff_text" });
    },
    wire() {
      el("cra-review").addEventListener("click", () => {
        const source = el("cra-source").value;
        if (source === "diff_text" && !el("cra-diff-text").value.trim()) {
          this.setStatus("Paste a diff first.", "error");
          return;
        }
        if (source === "worktree" && !el("cra-folder").value.trim()) {
          this.setStatus("Enter a repository folder first.", "error");
          return;
        }
        if (source === "worktree") rememberPath("cra-folder");
        this.run({
          button: el("cra-review"),
          key: "code-review-assist",
          busy: "Reviewing…",
          work: async () => {
            const data = await postJSON("/api/code-review-assist/review", {
              source,
              folder: el("cra-folder").value,
              against: el("cra-against").value || "HEAD",
              diff_text: el("cra-diff-text").value,
              engine: el("cra-engine").value,
              compute_device: el("cra-compute-device").value,
            });
            this.renderResult(data);
            return data;
          },
        });
      });
    },
    renderResult(data) {
      let html = "";
      if (data.diff_truncated) {
        html += `<p class="section-label">Diff was ${data.diff_char_count} characters -- truncated before review, some changes may not be reflected.</p>`;
      }
      html += `<p class="section-label">Commit message</p><pre class="text-block">${escapeHtml(data.commit_message)}</pre>`;
      html += `<p class="section-label">Review notes</p><p class="text-block">${escapeHtml(data.review_notes)}</p>`;
      el("cra-result").innerHTML = html;
    },
  }),

  "html-creator": new Panel({
    id: "html-creator",
    prefix: "htmlc",
    currentHtml: null,
    populate(data) {
      attachRecents("htmlc-folder");
      const mode = el("htmlc-mode");
      const sync = () => {
        const isLandingPage = mode.value === "landing_page";
        el("htmlc-prompt-field").hidden = !isLandingPage;
        el("htmlc-folder-field").hidden = isLandingPage;
      };
      mode.addEventListener("change", sync);
      sync();
      wireEngineAndDevice(el("htmlc-engine"), el("htmlc-compute-device"), data, { preferLargeModel: true });
      wireSamplePicker("htmlc-sample", data.samples, { "htmlc-mode": "mode", "htmlc-prompt": "prompt", "htmlc-folder": "folder" });
    },
    wire() {
      el("htmlc-generate").addEventListener("click", () => {
        const mode = el("htmlc-mode").value;
        if (mode === "landing_page" && !el("htmlc-prompt").value.trim()) {
          this.setStatus("Describe the page first.", "error");
          return;
        }
        if (mode === "document" && !el("htmlc-folder").value.trim()) {
          this.setStatus("Enter a folder first.", "error");
          return;
        }
        if (mode === "document") rememberPath("htmlc-folder");
        el("htmlc-download").hidden = true;
        this.run({
          button: el("htmlc-generate"),
          key: "html-creator",
          busy: "Generating…",
          work: async () => {
            const data = await postJSON("/api/html-creator/generate", {
              mode,
              prompt: el("htmlc-prompt").value,
              folder: el("htmlc-folder").value,
              engine: el("htmlc-engine").value,
              compute_device: el("htmlc-compute-device").value,
            });
            this.renderResult(data);
            return data;
          },
        });
      });
      el("htmlc-download").addEventListener("click", () => {
        if (!this.currentHtml) return;
        const url = URL.createObjectURL(new Blob([this.currentHtml], { type: "text/html" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = "generated.html";
        link.click();
        URL.revokeObjectURL(url);
      });
    },
    renderResult(data) {
      this.currentHtml = data.html;
      const container = el("htmlc-result");
      container.innerHTML = "";
      if (data.html_truncated) {
        container.insertAdjacentHTML("beforeend", '<p class="section-label">Output doesn\'t end with </html> -- it may have been cut off.</p>');
      }
      if (data.source_truncated) {
        container.insertAdjacentHTML("beforeend", `<p class="section-label">Source was ${data.source_char_count} characters -- truncated before generation, some content may not be reflected.</p>`);
      }
      const iframe = document.createElement("iframe");
      iframe.className = "htmlc-preview-frame";
      iframe.setAttribute("sandbox", "allow-scripts");
      iframe.title = "Generated page preview";
      iframe.srcdoc = data.html;
      container.appendChild(iframe);
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "View raw HTML";
      details.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "text-block";
      pre.textContent = data.html;
      details.appendChild(pre);
      container.appendChild(details);
      el("htmlc-download").hidden = false;
    },
  }),
};

// --- Home grid ----------------------------------------------------------------------

function renderBadges(container, demo) {
  container.innerHTML = "";
  if (demo.requires_dgpu) {
    const badge = document.createElement("span");
    badge.className = "badge badge-dgpu";
    badge.textContent = "Discrete GPU";
    badge.title =
      "The OpenVINO engine's model here needs a real discrete GPU with its own VRAM -- too large for an iGPU's or NPU's memory budget. The portable engine still runs everywhere.";
    container.appendChild(badge);
  }
  if (demo.status !== "available") {
    const badge = document.createElement("span");
    badge.className = "badge badge-planned";
    badge.textContent = "Coming soon";
    container.appendChild(badge);
  }
}

let lastLaunchButton = null;

function renderCards(demos) {
  const root = el("categories");
  root.innerHTML = "";
  const template = el("card-template");

  const byCategory = new Map();
  for (const demo of demos) {
    if (!byCategory.has(demo.category)) byCategory.set(demo.category, []);
    byCategory.get(demo.category).push(demo);
  }
  const orderedCategories = [
    ...CATEGORY_ORDER.filter((c) => byCategory.has(c)),
    ...[...byCategory.keys()].filter((c) => !CATEGORY_ORDER.includes(c)),
  ];

  for (const category of orderedCategories) {
    const block = document.createElement("section");
    block.className = "category-block";
    const heading = document.createElement("h2");
    heading.className = "category-heading";
    heading.textContent = category;
    block.appendChild(heading);

    const grid = document.createElement("div");
    grid.className = "card-grid";
    for (const demo of byCategory.get(category)) {
      const node = template.content.cloneNode(true);
      const card = node.querySelector(".card");
      card.dataset.id = demo.id;
      node.querySelector(".card-name").textContent = demo.name;
      node.querySelector(".card-tagline").textContent = demo.tagline;
      renderBadges(node.querySelector(".badges"), demo);
      const button = node.querySelector(".launch-btn");
      if (demo.status === "available" && PANELS[demo.id]) {
        const open = () => {
          lastLaunchButton = button;
          location.hash = `#/brick/${demo.id}`;
        };
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          open();
        });
        card.addEventListener("click", open);
      } else {
        card.classList.add("planned");
        button.remove();
      }
      grid.appendChild(node);
    }
    block.appendChild(grid);
    root.appendChild(block);
  }
}

// --- Routing --------------------------------------------------------------------

let currentPanel = null;

function route() {
  const match = location.hash.match(/^#\/brick\/([\w-]+)$/);
  const demo = match ? demoById(match[1]) : null;
  if (demo && demo.status === "available" && PANELS[demo.id]) showPanel(demo);
  else showHome();
}

async function showPanel(demo) {
  const panel = PANELS[demo.id];
  const switching = currentPanel !== panel;
  if (currentPanel && switching) currentPanel.leave();
  el("view-home").hidden = true;
  el("view-brick").hidden = false;
  for (const node of document.querySelectorAll(".brick-panel")) node.hidden = node.id !== `${panel.prefix}-panel`;
  el("panel-title").textContent = demo.name;
  el("panel-description").textContent = demo.description;
  renderBadges(el("panel-badges"), demo);
  document.title = `${demo.name} · Panther Lake AI Studio`;
  renderRunningStrip();
  window.scrollTo(0, 0);
  el("panel-title").focus({ preventScroll: true });
  if (switching) {
    currentPanel = panel;
    await panel.open();
  }
}

function showHome() {
  if (currentPanel) {
    currentPanel.leave();
    currentPanel = null;
  }
  el("view-brick").hidden = true;
  el("view-home").hidden = false;
  document.title = "Panther Lake AI Studio";
  renderRunningStrip();
  if (lastLaunchButton) {
    lastLaunchButton.focus({ preventScroll: true });
    lastLaunchButton = null;
  }
}

// --- Now running strip + card states ------------------------------------------------

// A brick with several stages (expense-extract's OCR/LLM, smart-city's feeds)
// has one entry per stage: stopping beats loading beats running (each says
// more than the next), and an error is shown only briefly after it happened
// -- the status snapshot keeps errors until the next run, and a chip that
// never goes away would just be noise.
function summarizeStatus(entries) {
  const stopping = entries.find((e) => e.phase === "stopping");
  if (stopping) return stopping;
  const loading = entries.find((e) => e.phase === "loading");
  if (loading) return loading;
  const running = entries.find((e) => e.phase === "running");
  if (running) return running;
  const nowSeconds = Date.now() / 1000;
  return entries.find((e) => e.phase === "error" && nowSeconds - e.at < 120) || null;
}

function renderRunningStrip() {
  const groups = new Map();
  for (const [key, entry] of Object.entries(STATUS.snapshot)) {
    const base = key.split(":")[0];
    if (!groups.has(base)) groups.set(base, []);
    groups.get(base).push(entry);
  }

  const chips = [];
  const phases = new Map();
  for (const [base, entries] of groups) {
    const status = summarizeStatus(entries);
    if (!status) continue;
    phases.set(base, status.phase);
    const demo = demoById(base);
    const devices = [...new Set(STATUS.active.filter((a) => a.demo_id === base).map((a) => a.device))];
    const meta = [status.phase, ...devices].join(" · ");
    const current = currentPanel && currentPanel.id === base ? " current" : "";
    chips.push(
      `<button type="button" class="running-chip phase-${status.phase}${current}" data-id="${escapeHtml(base)}" title="${escapeHtml(status.message || "")}">` +
        `<span class="chip-dot"></span><span class="chip-name">${escapeHtml(demo ? demo.name : base)}</span>` +
        `<span class="chip-meta">${escapeHtml(meta)}</span></button>`,
    );
  }
  const container = el("running-chips");
  const markup = chips.join("");
  if (container.innerHTML !== markup) {
    container.innerHTML = markup;
    for (const chip of container.querySelectorAll(".running-chip")) {
      chip.addEventListener("click", () => {
        location.hash = `#/brick/${chip.dataset.id}`;
      });
    }
  }
  el("running-strip").hidden = chips.length === 0;

  for (const card of document.querySelectorAll(".card[data-id]")) {
    const phase = phases.get(card.dataset.id);
    card.classList.toggle("is-running", phase === "running");
    card.classList.toggle("is-loading", phase === "loading" || phase === "stopping");
    card.querySelector(".card-state").textContent = PHASE_LABEL[phase] || "";
  }
}

async function pollStatus() {
  try {
    STATUS.snapshot = await fetchJSON("/api/status");
  } catch {
    return; // best-effort -- a missed poll just skips this tick
  }
  renderRunningStrip();
  if (currentPanel) currentPanel.onStatus();
}

// --- Telemetry --------------------------------------------------------------------

function matchGaugeKind(device) {
  const d = (device || "").toUpperCase();
  if (d === "CPU") return "cpu";
  // A specific GPU id ("GPU.0", "GPU.1", or bare "GPU" on a single-GPU
  // machine), so pinning a demo's stage to one physical GPU lights up only
  // that GPU's gauge.
  if (d.startsWith("GPU")) return d;
  if (d === "NPU") return "npu";
  return null; // "AUTO" or "cuda": picked internally by the runtime, not pinned to one gauge
}

function renderTelemetry(data) {
  STATUS.active = data.active || [];
  // data.active is a list, not a dict keyed by demo id: a demo like
  // expense-extract has two entries at once (one per stage, on two devices).
  const activeByKind = {};
  for (const info of STATUS.active) {
    const kind = matchGaugeKind(info.device);
    if (!kind) continue;
    const demo = demoById(info.demo_id);
    const baseName = demo ? demo.name : info.demo_id;
    activeByKind[kind] = info.stage_label ? `${baseName} (${info.stage_label})` : baseName;
  }

  const gauges = {
    cpu: { value: data.cpu_percent, name: null, selector: '.telemetry-gauge[data-device="cpu"]' },
    npu: { value: data.npu_percent, name: data.npu_name, selector: '.telemetry-gauge[data-device="npu"]' },
  };
  for (const gpu of data.gpus || []) {
    gauges[gpu.id] = { value: gpu.percent, name: gpu.name, selector: `.telemetry-gauge[data-gpu-id="${gpu.id}"]` };
  }

  for (const [kind, { value, name, selector }] of Object.entries(gauges)) {
    const gauge = document.querySelector(selector);
    if (!gauge) continue;
    const valueEl = gauge.querySelector(".telemetry-gauge-value");
    const fillEl = gauge.querySelector(".telemetry-bar-fill");
    const noteEl = gauge.querySelector(".telemetry-gauge-note");
    if (value === null || value === undefined) {
      valueEl.textContent = "N/A";
      fillEl.style.width = "0%";
      gauge.classList.add("unavailable");
    } else {
      valueEl.textContent = `${Math.round(value)}%`;
      fillEl.style.width = `${Math.min(value, 100)}%`;
      gauge.classList.remove("unavailable");
    }
    const activeLabel = activeByKind[kind];
    noteEl.textContent = activeLabel || name || "";
    gauge.classList.toggle("active-gauge", Boolean(activeLabel));
  }
  renderDeviceSummary(data);
  // The chips' device labels come from this poll, not the status one --
  // refresh them now rather than up to 1.5s later.
  renderRunningStrip();
}

let deviceSummaryDone = false;

function renderDeviceSummary(telemetry) {
  if (deviceSummaryDone) return;
  const parts = ["CPU", ...GPU_DEVICES.map((g) => shortGpuName(g.full_name))];
  if (telemetry.npu_percent !== null && telemetry.npu_percent !== undefined) parts.push("NPU");
  el("device-summary").textContent = `Inference devices: ${parts.join(" · ")}`;
  deviceSummaryDone = true;
}

async function pollTelemetry() {
  try {
    renderTelemetry(await fetchJSON("/api/telemetry"));
  } catch {
    // Best-effort panel -- ignore a transient failure and try again next tick.
  }
}

// One gauge per detected GPU; GPU_DEVICES is the source of truth for which
// gauges exist, /api/telemetry polls only fill in their values.
function initGpuGauges() {
  const template = el("telemetry-gpu-gauge-template");
  const container = document.querySelector(".telemetry-gpu-gauges");
  for (const gpu of GPU_DEVICES) {
    const gauge = template.content.cloneNode(true).querySelector(".telemetry-gauge");
    gauge.dataset.gpuId = gpu.id;
    if (GPU_DEVICES.length > 1) gauge.querySelector(".telemetry-gauge-label").textContent = gpu.id;
    container.appendChild(gauge);
  }
}

async function initTelemetry() {
  try {
    GPU_DEVICES = await fetchJSON("/api/system/gpu-devices");
  } catch {
    GPU_DEVICES = [];
  }
  initGpuGauges();
  pollTelemetry();
  setInterval(pollTelemetry, 2000);
}

// --- Activity log -------------------------------------------------------------------

function logDemoLabel(demoId) {
  const [baseId, stage] = demoId.split(":");
  const demo = demoById(baseId);
  const base = demo ? demo.name : baseId;
  return stage ? `${base} (${stage})` : base;
}

async function openLogViewer() {
  el("log-modal-overlay").hidden = false;
  el("log-modal-close").focus();
  const container = el("log-list");
  try {
    const entries = await fetchJSON("/api/logs?limit=100");
    if (!entries.length) {
      showPlaceholder(container, "No events yet -- launch a brick to see activity here.");
      return;
    }
    container.innerHTML = entries
      .slice()
      .reverse()
      .map(
        (entry) =>
          `<div class="log-entry${entry.phase === "error" ? " error" : ""}">` +
          `<span class="log-entry-time">${new Date(entry.at * 1000).toLocaleTimeString()}</span>` +
          `<span class="log-entry-demo">${escapeHtml(logDemoLabel(entry.demo_id))}</span>` +
          `<span class="log-entry-message">${escapeHtml(entry.message || entry.phase)}</span></div>`,
      )
      .join("");
  } catch (err) {
    showPlaceholder(container, `Error loading log: ${err.message}`);
  }
}

function closeLogViewer() {
  el("log-modal-overlay").hidden = true;
  el("log-open").focus();
}

// --- Init ------------------------------------------------------------------------

async function loadVersion() {
  try {
    const data = await fetchJSON("/api/version");
    el("app-version").textContent = `v${data.version}`;
  } catch {
    // Best-effort -- an empty footer label beats breaking page load over it.
  }
}

async function init() {
  DEMOS = await fetchJSON("/api/demos");
  renderCards(DEMOS);
  loadVersion();
  initTelemetry();

  for (const panel of Object.values(PANELS)) panel.wire();

  el("panel-back").addEventListener("click", () => {
    location.hash = "#/";
  });
  el("log-open").addEventListener("click", openLogViewer);
  el("log-modal-close").addEventListener("click", closeLogViewer);
  el("log-modal-overlay").addEventListener("click", (event) => {
    if (event.target === el("log-modal-overlay")) closeLogViewer();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!el("log-modal-overlay").hidden) closeLogViewer();
    else if (currentPanel && !["TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) location.hash = "#/";
  });

  await pollStatus();
  setInterval(pollStatus, 1500);
  window.addEventListener("hashchange", route);
  route();
}

init();
