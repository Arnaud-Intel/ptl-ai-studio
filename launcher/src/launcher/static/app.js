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

// The device a live-video model should default to: the integrated GPU, not
// AUTO. Measured on this machine, YOLO11s over street-camera frames: AUTO
// 33.6ms, iGPU 7.9ms, NPU 20.8ms, CPU 20.7ms -- identical detections, four
// times the latency. The discrete card is skipped on purpose: it carries a
// fixed ~18ms per-frame cost (the frame has to cross PCIe and come back)
// that no amount of extra compute pays back on a model this small.
function preferredRealtimeVisionDevice(openvinoDevices) {
  const available = GPU_DEVICES.filter((g) => (openvinoDevices || []).includes(g.id));
  const integrated = available.filter((g) => !(g.full_name || "").includes("dGPU"));
  const pick = integrated[0] || available[0];
  return pick ? pick.id : "AUTO";
}

// --- Control wiring ---------------------------------------------------------------

// Every brick whose OCR stage is screen-ocr's OpenVINO engine runs the same
// 7B vision-language model, and that model's NPU compile fails on this
// hardware -- reliably, with a compiler error from deep inside OpenVINO
// (see screen-ocr's README). Offer the device, but not silently.
const OCR_MODEL_UNSUPPORTED = { NPU: "this OCR model doesn't compile for the NPU" };

// One engine <select> + one compute-device <select>, kept consistent: the
// OpenVINO option is only enabled when this brick actually has a device, the
// device list follows the chosen engine (AUTO + every OpenVINO device, or the
// portable engine's fixed choices), and GPU ids get their friendly names.
function wireEngineAndDevice(engineSelect, deviceSelect, data, options = {}) {
  const {
    portableDevices = ["cpu"],
    preferLargeModel = false,
    preferRealtimeVision = false,
    onChange = null,
    unsupported = {},
  } = options;
  const openvinoDevices = data.openvino_devices || [];
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = openvinoDevices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  if (!hasOpenvino) openvinoOption.textContent = `${openvinoOption.textContent} -- not installed for this brick`;
  if (hasOpenvino) engineSelect.value = "openvino";

  // What the user last chose, per engine. Switching engine rebuilds the
  // list (the two engines offer different devices), and without this a
  // deliberate "run it on the NPU" silently became AUTO on the way back --
  // which then picks its own device, and the run lands somewhere the user
  // never asked for.
  const chosenPerEngine = {};
  deviceSelect.addEventListener("change", () => {
    chosenPerEngine[engineSelect.value] = deviceSelect.value;
  });

  const fill = () => {
    const isOpenvino = engineSelect.value === "openvino";
    const values = isOpenvino ? ["AUTO", ...openvinoDevices] : portableDevices;
    fillSelect(deviceSelect, values.map((value) => ({ value, label: deviceLabel(value) })));
    // A device this brick's model provably can't use is shown but disabled,
    // with the reason in the label -- offering it silently is how someone
    // ends up staring at a compiler error from deep inside OpenVINO.
    if (isOpenvino) {
      for (const opt of deviceSelect.options) {
        const reason = unsupported[opt.value] || (data.openvino_unsupported || {})[opt.value];
        if (!reason) continue;
        opt.disabled = true;
        opt.textContent = `${opt.textContent} -- ${reason}`;
      }
      if (deviceSelect.selectedOptions[0]?.disabled) deviceSelect.value = "AUTO";
    }
    if (isOpenvino && preferLargeModel) deviceSelect.value = preferredLargeModelDevice(openvinoDevices);
    if (isOpenvino && preferRealtimeVision) deviceSelect.value = preferredRealtimeVisionDevice(openvinoDevices);
    const chosen = chosenPerEngine[engineSelect.value];
    if (chosen && [...deviceSelect.options].some((o) => o.value === chosen && !o.disabled)) {
      deviceSelect.value = chosen;
    }
    if (!deviceSelect.selectedOptions.length || deviceSelect.selectedOptions[0].disabled) {
      deviceSelect.value = [...deviceSelect.options].find((o) => !o.disabled)?.value || "";
    }
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
function renderDemoAssets(container, assets) {
  const grid = document.createElement("div");
  grid.className = "demo-asset-grid";
  for (const asset of assets || []) {
    const link = document.createElement("a");
    link.href = asset.url;
    link.target = "_blank";
    link.rel = "noopener";
    link.className = "demo-asset";
    if (asset.image) {
      const image = document.createElement("img");
      image.src = asset.url;
      image.alt = `Preview of fictional sample ${asset.name}`;
      image.loading = "lazy";
      link.appendChild(image);
    }
    const label = document.createElement("span");
    label.textContent = asset.name;
    link.appendChild(label);
    grid.appendChild(link);
  }
  container.appendChild(grid);
}

function wireSamplePicker(pickerId, samples, fieldMap, onSelect = null) {
  const picker = el(pickerId);
  if (!picker) return;
  while (picker.options.length > 1) picker.remove(1);
  for (const sample of samples || []) {
    picker.appendChild(option(sample.name, `${sample.name} — ${sample.description}`));
  }
  picker.disabled = !(samples || []).length;
  const panel = picker.closest(".brick-panel");
  let details = el(`${pickerId}-details`);
  if (!details) {
    details = document.createElement("div");
    details.id = `${pickerId}-details`;
    details.className = "demo-sample-details";
    details.hidden = true;
    // Keep the run action above the gallery, even for the complete receipt pack.
    (panel?.querySelector(".expense-review") || panel?.querySelector(".run-row") || panel?.querySelector(".controls") || picker).after(details);
  }
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
      details.replaceChildren();
      details.hidden = false;
      const title = document.createElement("h3");
      title.textContent = sample.name;
      const text = document.createElement("p");
      text.textContent = `${sample.description} ${sample.next_step || ""}`;
      details.append(title, text);
      renderDemoAssets(details, sample.assets);
      if (onSelect) onSelect(sample);
    } else {
      details.hidden = true;
      if (onSelect) onSelect(null);
    }
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
        if (data.demo_guide) {
          const root = el(`${this.prefix}-panel`);
          const guide = document.createElement("details");
          guide.className = "demo-guide";
          const summary = document.createElement("summary");
          summary.textContent = `Demo guide · ${data.demo_guide.title}`;
          const content = document.createElement("p");
          content.textContent = data.demo_guide.text;
          guide.append(summary, content);
          renderDemoAssets(guide, data.demo_guide.assets);
          root?.prepend(guide);
        }
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


// --- smart-city feed cards ------------------------------------------------
// One card per feed, added and removed by the user. Each one carries its
// own engine, model, chip and source, because the pipeline now loads one
// detector per distinct (engine, device, model) -- so two cards can run
// genuinely different backends at once.

// Each engine ships exactly one model; "Custom path" is what makes the
// control more than decoration, and it is the only way to reach the
// pipeline's model_path from the UI.
const FEED_MODELS = {
  portable: [{ value: "", label: "DETR-ResNet-50 (built-in)" }],
  openvino: [{ value: "", label: "YOLO11s INT8 (built-in)" }],
};
const CUSTOM_MODEL = "__custom__";

let feedDevicesData = null;  // the /devices payload, needed by every new card
let feedCounter = 0;

function feedCards() {
  return [...document.querySelectorAll("#smartcity-feed-list .feed-card")];
}

function renumberFeeds() {
  const cards = feedCards();
  cards.forEach((card, i) => {
    card.querySelector(".feed-card-title").textContent = `Feed ${i + 1}`;
  });
  el("smartcity-feed-empty").hidden = cards.length > 0;
}

// The value a card contributes to the start request: whichever source box
// its type is showing, plus how that feed wants to be run.
function feedCardValue(card) {
  const type = card.querySelector(".feed-type").value;
  const path = type === "url"
    ? card.querySelector(".feed-url").value.trim()
    : card.querySelector(".feed-path").value.trim();
  const model = card.querySelector(".feed-model").value;
  return {
    path,
    engine: card.querySelector(".feed-engine").value,
    compute_device: card.querySelector(".feed-device").value,
    model_path: model === CUSTOM_MODEL ? card.querySelector(".feed-model-path").value.trim() || null : null,
  };
}

async function uploadFeedVideo(card, file) {
  const zone = card.querySelector(".dropzone");
  const text = zone.querySelector(".dropzone-text");
  const body = new FormData();
  body.append("file", file);
  zone.classList.add("is-busy");
  text.textContent = `Copying ${file.name}...`;
  try {
    const res = await fetch("/api/smart-city-monitor/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `upload failed (${res.status})`);
    card.querySelector(".feed-path").value = data.path;
    text.textContent = `${data.name} -- ${(data.bytes / 1048576).toFixed(1)} MB, ready`;
  } catch (err) {
    text.textContent = `Couldn't take that file: ${err.message}`;
  } finally {
    zone.classList.remove("is-busy");
  }
}

// A <select>'s options grouped under an <optgroup> per item.group, after
// whatever placeholder option it already leads with. The curated cameras
// use it to separate YouTube from everything else -- which matters the day
// YouTube refuses the whole network, and only the other section still works.
function fillGroupedSelect(select, items, valueOf, labelOf) {
  const placeholder = select.options[0] && select.options[0].value === "" ? select.options[0] : null;
  const groups = new Map();
  for (const item of items) {
    const key = item.group || "Other";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  const sections = [...groups].map(([label, entries]) => {
    const section = document.createElement("optgroup");
    section.label = label;
    for (const entry of entries) section.appendChild(option(valueOf(entry), labelOf(entry)));
    return section;
  });
  select.replaceChildren(...(placeholder ? [placeholder] : []), ...sections);
}

function wireFeedCard(card) {
  const engine = card.querySelector(".feed-engine");
  const device = card.querySelector(".feed-device");
  const model = card.querySelector(".feed-model");
  const modelPathField = card.querySelector(".feed-model-path-field");
  const type = card.querySelector(".feed-type");

  // The same helper every other panel uses, just scoped to this card's
  // pair of selects -- so a card gets the engine/device rules for free.
  if (feedDevicesData) wireEngineAndDevice(engine, device, feedDevicesData, { preferRealtimeVision: true });

  const fillModels = () => {
    const options = [...(FEED_MODELS[engine.value] || []), { value: CUSTOM_MODEL, label: "Custom path..." }];
    fillSelect(model, options);
    modelPathField.hidden = true;
  };
  engine.addEventListener("change", fillModels);
  model.addEventListener("change", () => {
    modelPathField.hidden = model.value !== CUSTOM_MODEL;
  });
  fillModels();

  // The curated cameras, offered per feed so a URL can be picked as well as
  // typed. Only the single-camera samples: a sample naming several feeds
  // ("two cities, two chips") describes a whole set-up, not one box, and
  // belongs to the add-a-feed picker instead.
  const urlSample = card.querySelector(".feed-url-sample");
  const cameras = (feedDevicesData?.samples || []).filter((entry) => !entry.feeds.includes("\n"));
  fillGroupedSelect(urlSample, cameras, (c) => c.feeds, (c) => `${c.name} -- ${c.description}`);
  urlSample.disabled = !cameras.length;
  urlSample.addEventListener("change", () => {
    if (urlSample.value) card.querySelector(".feed-url").value = urlSample.value;
    // Reset: once it's in the box the URL is the truth, and a stale
    // selection would keep claiming a camera the user has since edited.
    urlSample.value = "";
  });

  const showSource = () => {
    const isUrl = type.value === "url";
    card.querySelector(".feed-source-file").hidden = isUrl;
    card.querySelector(".feed-source-url").hidden = !isUrl;
  };
  type.addEventListener("change", showSource);
  showSource();

  // Drag-and-drop, and the same zone as a click-to-browse. A browser never
  // tells a page where a dropped file lives, so the bytes are copied to the
  // launcher and the path it landed at goes in the box -- typing a path
  // straight in stays the way to use a big file where it already is.
  const zone = card.querySelector(".dropzone");
  const fileInput = card.querySelector(".feed-file-input");
  zone.addEventListener("click", () => fileInput.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) uploadFeedVideo(card, fileInput.files[0]);
  });
  for (const name of ["dragenter", "dragover"]) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("is-over");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    zone.addEventListener(name, () => zone.classList.remove("is-over"));
  }
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) uploadFeedVideo(card, file);
  });

  card.querySelector(".feed-remove").addEventListener("click", () => {
    card.remove();
    renumberFeeds();
  });
}

// Point a device select at `wanted` without ever leaving it blank. A
// preset names a device the machine may not enumerate under that exact id:
// a sample pinned to "GPU" has to land on "GPU.0" on a two-GPU box, and on
// nothing at all if there is no GPU -- in which case AUTO is the honest
// answer, not an empty select that posts an empty device.
function selectDevice(select, wanted) {
  const options = [...select.options].filter((o) => !o.disabled).map((o) => o.value);
  const exact = options.find((value) => value === wanted);
  const prefixed = options.find((value) => value.startsWith(`${wanted}.`));
  select.value = exact || prefixed || (options.includes("AUTO") ? "AUTO" : options[0] || "");
  return select.value;
}

// `preset` fills a new card in: {type, path, engine, device, name}.
function addFeedCard(preset = {}) {
  const template = el("smartcity-feed-template");
  const card = template.content.firstElementChild.cloneNode(true);
  card.dataset.key = `card-${++feedCounter}`;
  el("smartcity-feed-list").appendChild(card);
  wireFeedCard(card);

  if (preset.engine) {
    card.querySelector(".feed-engine").value = preset.engine;
    card.querySelector(".feed-engine").dispatchEvent(new Event("change"));
  }
  if (preset.device) selectDevice(card.querySelector(".feed-device"), preset.device);
  if (preset.type) {
    card.querySelector(".feed-type").value = preset.type;
    card.querySelector(".feed-type").dispatchEvent(new Event("change"));
  }
  if (preset.path) {
    const target = preset.type === "url" ? ".feed-url" : ".feed-path";
    card.querySelector(target).value = preset.path;
  }
  renumberFeeds();
  return card;
}

// Sound tags for the voice models that understand them. Clicking one
// drops it at the cursor rather than at the end, because where a laugh
// falls in a sentence is the whole point of having it.
function renderTagButtons(tags) {
  const row = el("voice-tags");
  if (!row) return;
  const markup = (tags || [])
    .map((tag) => `<button type="button" class="tag-btn" data-tag="${escapeHtml(tag)}" disabled>${escapeHtml(tag)}</button>`)
    .join("");
  if (row.innerHTML === markup) return;
  row.innerHTML = markup;
  for (const button of row.querySelectorAll(".tag-btn")) {
    button.addEventListener("click", () => {
      const box = el("voice-text");
      const tag = button.dataset.tag;
      const start = box.selectionStart ?? box.value.length;
      const end = box.selectionEnd ?? box.value.length;
      const before = box.value.slice(0, start);
      const after = box.value.slice(end);
      // Pad only where there isn't already a space, so dropping a tag
      // mid-sentence doesn't leave a double gap behind it.
      const lead = before && !/\s$/.test(before) ? " " : "";
      const trail = !after || /^\s/.test(after) ? "" : " ";
      const inserted = lead + tag + trail;
      box.value = before + inserted + after;
      box.focus();
      box.selectionStart = box.selectionEnd = start + inserted.length;
    });
  }
}

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
      wireEngineAndDevice(el("expx-ocr-engine"), el("expx-ocr-device"), data, { unsupported: OCR_MODEL_UNSUPPORTED });
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
      ExpenseReviewUI.refresh();
    },
    body() {
      const folder = el("expx-folder").value.trim();
      if (!folder) throw new Error("Enter a folder of receipt photos first.");
      if (!ExpenseReviewUI.confirmNewBatch()) throw new Error("Current review kept. Export it before starting another batch.");
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
      ExpenseReviewUI.refresh(true);
      showPlaceholder(el("expx-transcript"), "Each receipt's vendor, date, amount, and category will appear here as it is structured.");
    },
    onRunning() { ExpenseReviewUI.refresh(); },
    onMessage(message) {
      const box = el("expx-transcript");
      if (message.type === "ocr_progress") {
        appendLine(box, "line-note", `Reading receipt ${message.index}/${message.total}: ${message.file}`);
      } else if (message.type === "structured") {
        const line = message.line;
        if (line.error) {
          appendLine(box, "line-answer", `${line.source_file}: skipped (${line.error})`);
        } else {
          const amount = line.amount !== null && line.amount !== undefined ? `${line.currency || "Unknown currency"} ${line.amount}` : "Unknown amount";
          const review = line.needs_review ? ` -- Needs review: ${(line.review_reasons || []).join("; ")}` : " -- Fields validated; verify against receipt";
          appendLine(box, "line-answer", `${line.source_file}: ${line.vendor || "?"} -- ${line.date || "?"} -- ${amount} -- ${line.category}${review}`);
        }
      } else if (message.type === "done") {
        this.setRunning(false);
        ExpenseReviewUI.refresh();
        this.setStatus(`Extraction finished — review and approve receipts below.`, "live");
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
      wireEngineAndDevice(el("recall-ocr-engine"), el("recall-ocr-device"), data, { unsupported: OCR_MODEL_UNSUPPORTED });
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
      wireEngineAndDevice(el("objdet-engine"), el("objdet-compute-device"), data, { preferRealtimeVision: true });
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
    // A disabled <fieldset> disables every card inside it, however many.
    controls: ["smartcity-feed-set", "smartcity-loop"],
    feeds: [],
    populate(data) {
      feedDevicesData = data;  // every card added later needs the device lists
      el("smartcity-add-feed").addEventListener("click", () => addFeedCard({ type: "file" }));
      el("smartcity-feed-picker").addEventListener("change", () => this.attachVideo());

      // A sample is a set of feeds, not a value for one box: each of its
      // lines ("url" or "url|DEVICE") becomes its own card, so "two cities,
      // two chips" arrives as two cards already pinned to their chips.
      const picker = el("smartcity-sample");
      const samples = data.samples || [];
      fillGroupedSelect(picker, samples, (entry) => entry.name, (entry) => `${entry.name} -- ${entry.description}`);
      picker.disabled = !samples.length;
      picker.onchange = () => {
        const sample = samples.find((entry) => entry.name === picker.value);
        picker.value = "";
        if (!sample) return;
        for (const line of sample.feeds.split("\n").map((l) => l.trim()).filter(Boolean)) {
          const pipe = line.lastIndexOf("|");
          addFeedCard({
            type: "url",
            path: pipe === -1 ? line : line.slice(0, pipe),
            device: pipe === -1 ? null : line.slice(pipe + 1).trim(),
          });
        }
      };

      if (!feedCards().length) addFeedCard({ type: "url" });  // start with one to fill in
      renumberFeeds();
    },
    async rehydrate() {
      // The feed list lives on the server (a run may predate this visit):
      // pick it up from the counts route, then the usual running check. The
      // route reports each feed's engine, chip and source, so the cards can
      // be rebuilt exactly as they were rather than left blank under a run.
      try {
        const data = await fetchJSON("/api/smart-city-monitor/counts");
        this.feeds = data.feeds || [];
        if (this.feeds.length) {
          el("smartcity-feed-list").innerHTML = "";
          for (const feed of this.feeds) {
            addFeedCard({
              type: feed.source_type || "url",
              path: feed.path,
              engine: feed.engine,
              device: feed.compute_device,
            });
          }
        }
      } catch {
        this.feeds = [];
      }
      await StreamPanel.prototype.rehydrate.call(this);
    },
    body() {
      const cards = feedCards();
      if (!cards.length) throw new Error("Add at least one feed.");
      const feeds = cards.map(feedCardValue);
      const blank = feeds.findIndex((feed) => !feed.path);
      if (blank !== -1) throw new Error(`Feed ${blank + 1} has no source yet -- choose a video file or enter a URL.`);
      // No run-wide engine or device: every card carries its own, and the
      // route only falls back per feed if one is somehow left unset.
      return { feeds, loop: el("smartcity-loop").checked };
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
      const counts = el("smartcity-counts");
      if (!isRunning) {
        el("smartcity-picker-row").hidden = true;
        picker.innerHTML = "";
        showPlaceholder(counts, "Per-minute counts will appear here once running, one block per feed.");
        for (const card of feedCards()) card.querySelector(".feed-card-counts").textContent = "";
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
          // One block per feed rather than one combined total. Two cameras
          // summed tell you nothing useful -- 90/min across Tokyo and
          // Dublin is not a number about anywhere -- and which feed is busy
          // is the thing the per-feed devices were set up to show.
          const perFeed = data.snapshot.per_feed_last_60s || {};
          counts.innerHTML =
            this.feeds
              .map((feed) => {
                // A class stays in the map with a count of 0 once its last
                // sighting ages out of the 60s window; "Buses: 0/min" is noise.
                const rows = Object.entries(perFeed[feed.feed_id] || {}).filter(([, n]) => n > 0);
                const body = rows.length
                  ? rows.map(([label, count]) => statRow(label, `${count}/min`)).join("")
                  : '<p class="placeholder">Nothing counted yet.</p>';
                return (
                  `<div class="feed-counts"><p class="feed-counts-head">${escapeHtml(feed.name)}` +
                  `<span class="feed-counts-device">${escapeHtml(feed.compute_device)}</span></p>${body}</div>`
                );
              })
              .join("") || '<p class="placeholder">No feeds running.</p>';
          // Each feed's numbers go on its own card -- the thing you set up
          // is the thing you read the result from.
          const cards = feedCards();
          this.feeds.forEach((feed, i) => {
            const counts = (data.snapshot.per_feed_last_60s || {})[feed.feed_id] || {};
            const parts = Object.entries(counts)
              .filter(([, count]) => count > 0)
              .map(([label, count]) => `${label}: ${count}/min`)
              .join(" · ") || "nothing counted yet";
            const card = cards[i];
            if (card) card.querySelector(".feed-card-counts").textContent = parts;
          });
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
      this.sampleImageUrl = null;
      wireSamplePicker("ocr-sample", data.samples, { "ocr-source": "source" }, (sample) => {
        this.sampleImageUrl = sample?.image_url || null;
      });
      const source = el("ocr-source");
      const sync = () => {
        const isUpload = source.value === "upload";
        el("ocr-source-device-field").hidden = isUpload || source.value === "sample";
        el("ocr-upload-field").hidden = !isUpload;
      };
      wireVideoSource(source, el("ocr-source-device"), data);
      source.addEventListener("change", sync);
      sync();
      const translate = el("ocr-translate");
      wireEngineAndDevice(el("ocr-engine"), el("ocr-compute-device"), data, {
        unsupported: OCR_MODEL_UNSUPPORTED,
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
        if (source === "sample" && !this.sampleImageUrl) {
          this.setStatus("Choose a bundled sample first.", "error");
          return;
        }
        const sampleImageUrl = this.sampleImageUrl;
        this.run({
          button: el("ocr-extract"),
          key: "screen-ocr",
          busy: source === "upload" ? "Uploading and reading…" : "Capturing and reading…",
          work: async () => {
            let data;
            if (source === "upload" || source === "sample") {
              const form = new FormData();
              if (source === "sample") {
                const response = await fetch(sampleImageUrl);
                if (!response.ok) throw new Error("Could not load the bundled sample image.");
                form.append("file", await response.blob(), "fictional-demo.png");
              } else {
                form.append("file", el("ocr-upload").files[0]);
              }
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
      this.wireModelChoice();
      const tau = el("voice-tau");
      const readout = el("voice-tau-value");
      tau.addEventListener("input", () => {
        readout.value = Number(tau.value).toFixed(2);
      });
      wireSamplePicker("voice-sample", data.samples, { "voice-text": "text" });
    },
    // What each model is, in the terms someone choosing between them cares
    // about. The similarity figures are measured (ECAPA-TDNN, against a
    // real human reference clip), not marketing.
    MODEL_INFO: {
      chatterbox: {
        engines: ["portable"],
        styles: false,
        tags: ["[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]"],
        help: "Chatterbox Turbo (350 MB, MIT). Reproduces a real speaker much more closely -- 0.65 " +
              "measured similarity to a human reference against OpenVoice's 0.29 -- and takes the sound " +
              "tags below. Runs on the CPU only, at roughly 2.5x realtime.",
      },
      openvoice: {
        engines: ["portable", "openvino"],
        styles: true,
        tags: [],
        help: "OpenVoice (MIT). Re-colors the tone of one of nine fixed base voices, so it captures " +
              "timbre rather than a person's full identity -- but it is the model that runs on the NPU " +
              "and iGPU, and it is faster (about 1.5x realtime).",
      },
    },
    modelInfo() {
      return this.MODEL_INFO[el("voice-model").value] || this.MODEL_INFO.chatterbox;
    },
    wireModelChoice() {
      const model = el("voice-model");
      const engine = el("voice-engine");
      const apply = () => {
        const info = this.modelInfo();
        el("voice-model-help").textContent = info.help;
        // A model with one engine has nothing to choose: hide the pair
        // rather than offering a control whose only value is forced.
        const oneEngine = info.engines.length === 1;
        el("voice-engine-field").hidden = oneEngine;
        el("voice-device-field").hidden = oneEngine;
        if (oneEngine) engine.value = info.engines[0];
        for (const option of engine.options) option.disabled = !info.engines.includes(option.value);
        el("voice-style-field").hidden = !info.styles;
        el("voice-tau-field").hidden = !info.styles;
        el("voice-tags-field").hidden = !info.tags.length;
        renderTagButtons(info.tags);
      };
      model.addEventListener("change", apply);
      apply();
    },
    async rehydrate() {
      try {
        const status = await fetchJSON("/api/voice-clone-studio/status");
        if (status.model) el("voice-model").value = status.model;
        el("voice-model").dispatchEvent(new Event("change"));
        this.setEnrolled(status.enrolled);
      } catch {
        this.setEnrolled(false);
      }
    },
    setEnrolled(enrolled) {
      for (const id of ["voice-text", "voice-sample", "voice-style", "voice-tau", "voice-synthesize"]) {
        el(id).disabled = !enrolled;
      }
      for (const button of document.querySelectorAll("#voice-tags .tag-btn")) button.disabled = !enrolled;
      if (enrolled) paintStatus(el("voice-enroll-status"), "Voice enrolled -- ready to speak", "live");
    },
    wire() {
      el("voice-enroll").addEventListener("click", () => {
        const source = el("voice-source").value;
        const model = el("voice-model").value;
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
              form.append("model", model);
              await fetchJSON("/api/voice-clone-studio/enroll-upload", { method: "POST", body: form });
            } else {
              await postJSON("/api/voice-clone-studio/enroll-record", {
                seconds: Number(el("voice-record-seconds").value) || 10,
                engine,
                compute_device: computeDevice,
                model,
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
              // Sent only when the model has them: the API refuses a
              // non-default style on a model with no styles, rather than
              // quietly ignoring what was asked for.
              body: JSON.stringify(
                this.modelInfo().styles
                  ? { text, style: el("voice-style").value, tau: Number(el("voice-tau").value) }
                  : { text },
              ),
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
        if (demo.status === "available") {
          console.error(
            `[panther-lake] "${demo.id}" is available on the server but has no panel in app.js -- ` +
              "showing it as unavailable. (A stale cached app.js will do this; try a hard reload.)",
          );
        }
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
  if (!demo || demo.status !== "available" || !PANELS[demo.id]) {
    if (match && !demo) console.error(`[panther-lake] no demo called "${match[1]}" -- going back to the grid.`);
    showHome();
    return;
  }
  // Never fail silently: a panel that can't open should say so somewhere,
  // rather than leaving a click looking like it did nothing.
  showPanel(demo).catch((err) => console.error(`[panther-lake] couldn't open "${demo.id}":`, err));
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

// The same chips render in the full strip and in the compact header, so a
// running demo is one click away whether or not you've scrolled.
function fillChips(containerId, markup) {
  const container = el(containerId);
  if (!container || container.innerHTML === markup) return;
  container.innerHTML = markup;
  for (const chip of container.querySelectorAll(".running-chip")) {
    chip.addEventListener("click", () => {
      location.hash = `#/brick/${chip.dataset.id}`;
    });
  }
}

// Show the compact header exactly once the real one has left the viewport.
// Asking the element where it is beats caching a scroll offset: it stays
// right when the running strip appears, the window resizes, or a wrapped
// header row changes height.
let compactFrame = 0;

function updateCompactBar() {
  const bar = el("compact-bar");
  const topbar = document.querySelector(".topbar");
  if (!bar || !topbar) return;
  const strip = el("running-strip");
  const anchor = strip && !strip.hidden ? strip : topbar;
  bar.classList.toggle("is-visible", anchor.getBoundingClientRect().bottom <= 0);
}

function wireCompactBar() {
  const onScroll = () => {
    if (compactFrame) return;  // at most one measure per frame
    compactFrame = requestAnimationFrame(() => {
      compactFrame = 0;
      updateCompactBar();
    });
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
  updateCompactBar();
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
  const markup = chips.join("");
  fillChips("running-chips", markup);
  fillChips("compact-chips", markup);  // the compact header shows the same set
  el("running-strip").hidden = chips.length === 0;
  updateCompactBar();  // the strip appearing/disappearing moves what we scroll past

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
  // A list per device, not one label: two demos (or two stages, or two
  // smart-city feeds) can be pinned to the same chip at once, and the old
  // single-slot version let whichever came last silently win -- so a
  // shared GPU looked exactly like a GPU with one demo on it.
  const activeByKind = {};
  for (const info of STATUS.active) {
    const kind = matchGaugeKind(info.device);
    if (!kind) continue;
    const demo = demoById(info.demo_id);
    const baseName = demo ? demo.name : info.demo_id;
    (activeByKind[kind] ||= []).push(info.stage_label ? `${baseName} (${info.stage_label})` : baseName);
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
    const sharing = activeByKind[kind] || [];
    // The note line is narrow, so lead with the count when a chip is
    // shared -- that is the fact worth noticing -- and put the full list
    // on the tooltip, which survives the ellipsis.
    noteEl.textContent = sharing.length > 1 ? `${sharing.length} demos: ${sharing.join(", ")}` : sharing[0] || name || "";
    if (sharing.length) noteEl.title = sharing.join(", ");
    else noteEl.removeAttribute("title");
    gauge.classList.toggle("active-gauge", sharing.length > 0);
    gauge.classList.toggle("shared-gauge", sharing.length > 1);
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
  el("device-summary").textContent = `Monitored hardware: ${parts.join(" · ")}. Inference support depends on the selected engine.`;
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
  setInterval(pollTelemetry, 600);
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
          `<span class="log-entry-time">${new Date(entry.at * 1000).toLocaleString()}</span>` +
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
    const label = el("app-version");
    label.textContent = `v${data.version}`;
    // Newer code sitting on disk unstarted is the single most confusing
    // state this app has: the page reloads (static files are read per
    // request) while the Python behind it stays as it was.
    if (data.restart_needed) {
      label.textContent = `v${data.version} -- restart to load v${data.on_disk}`;
      label.classList.add("version-stale");
      label.title = `This launcher started on v${data.version}. v${data.on_disk} is on disk; restart it to run that.`;
    } else {
      label.classList.remove("version-stale");
      label.removeAttribute("title");
    }
  } catch {
    // Best-effort -- an empty footer label beats breaking page load over it.
  }
}

async function init() {
  DEMOS = await fetchJSON("/api/demos");
  renderCards(DEMOS);
  loadVersion();
  initTelemetry();

  // Navigation is wired before the panels, and each panel independently:
  // one brick missing an element must not cost the whole app its routing.
  // (Wired the other way round, a single throw in wire() left every card
  // silently doing nothing when clicked, with no clue as to why.)
  window.addEventListener("hashchange", route);
  wireCompactBar();

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

  for (const panel of Object.values(PANELS)) {
    try {
      panel.wire();
    } catch (err) {
      console.error(`[panther-lake] "${panel.id}" failed to wire its controls -- its panel may not work:`, err);
    }
  }

  await pollStatus();
  setInterval(pollStatus, 1500);
  route();
}

init();
