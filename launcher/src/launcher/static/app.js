const CATEGORY_ORDER = ["Speech", "Vision", "Text", "Productivity", "Audio"];

const el = (id) => document.getElementById(id);

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.error || `${url} failed (${res.status})`);
  }
  return body;
}

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
      node.querySelector(".category-tag").textContent = demo.category;

      const pill = node.querySelector(".status-pill");
      pill.textContent = demo.status === "available" ? "Available" : "Planned";
      pill.classList.add(demo.status === "available" ? "available" : "planned");

      node.querySelector(".card-name").textContent = demo.name;
      node.querySelector(".card-tagline").textContent = demo.tagline;
      node.querySelector(".card-description").textContent = demo.description;

      const badges = node.querySelector(".engine-badges");
      for (const engine of demo.engines) {
        const span = document.createElement("span");
        span.className = "engine-badge";
        span.textContent = engine;
        badges.appendChild(span);
      }
      if (demo.requires_dgpu) {
        const span = document.createElement("span");
        span.className = "engine-badge requires-dgpu";
        span.textContent = "Discrete GPU";
        span.title = "The openvino engine's model here needs a real discrete GPU with its own VRAM -- too large for an iGPU's or NPU's memory budget. The portable engine still runs everywhere.";
        badges.appendChild(span);
      }

      const btn = node.querySelector(".launch-btn");
      if (demo.status === "available") {
        btn.addEventListener("click", () => openDemo(demo));
      } else {
        btn.textContent = "Coming soon";
        btn.disabled = true;
        card.addEventListener("click", () => openPlaceholder(demo));
      }

      grid.appendChild(node);
    }

    block.appendChild(grid);
    root.appendChild(block);
  }
}

function openPlaceholder(demo) {
  el("placeholder-text").textContent = `${demo.name}: ${demo.description}`;
  el("placeholder-overlay").classList.remove("hidden");
}

function openDemo(demo) {
  if (demo.id === "live-translation") {
    openLiveTranslation();
  } else if (demo.id === "doc-qa") {
    openDocQA();
  } else if (demo.id === "object-detection") {
    openObjectDetection();
  } else if (demo.id === "screen-ocr") {
    openScreenOcr();
  } else if (demo.id === "meeting-notes") {
    openMeetingNotes();
  } else if (demo.id === "webcam-effects") {
    openWebcamEffects();
  } else if (demo.id === "voice-clone-studio") {
    openVoiceCloneStudio();
  } else if (demo.id === "voice-assistant") {
    openVoiceAssistant();
  } else if (demo.id === "expense-extract") {
    openExpenseExtract();
  } else if (demo.id === "smart-recall") {
    openRecall();
  } else if (demo.id === "code-review-assist") {
    openCodeReviewAssist();
  } else if (demo.id === "html-creator") {
    openHtmlCreator();
  }
}

// --- Live translation demo ---

let ws = null;
let running = false;

async function openLiveTranslation() {
  el("modal-overlay").classList.remove("hidden");
  await populateLiveTranslationDevices();
  connectWebSocket();
}

async function populateLiveTranslationDevices() {
  const data = await fetchJSON("/api/live-translation/devices");

  const audioSelect = el("ctl-audio-device");
  audioSelect.innerHTML = "";
  const defaultOpt = document.createElement("option");
  defaultOpt.value = "";
  defaultOpt.textContent = "Default device";
  audioSelect.appendChild(defaultOpt);

  const sourceSelect = el("ctl-source");
  const fillAudioDevices = () => {
    const names = sourceSelect.value === "mic" ? data.microphones : data.speakers;
    audioSelect.innerHTML = "";
    audioSelect.appendChild(defaultOpt.cloneNode(true));
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      audioSelect.appendChild(opt);
    }
  };
  sourceSelect.onchange = fillAudioDevices;
  fillAudioDevices();

  const engineSelect = el("ctl-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("ctl-compute-device");
  // Intel only publishes pre-converted OpenVINO Whisper repos for
  // tiny/base/medium/large-v3 -- there's no whisper-small-fp16-ov, so
  // "small" throws a backend error under the OpenVINO engine. Disable it
  // in the <select> itself (same pattern as the OpenVINO engine option
  // being disabled when no device is available) rather than just
  // resetting the default, since nothing else stops a manual pick.
  const modelSelect = el("ctl-model");
  const smallOption = modelSelect.querySelector('option[value="small"]');
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino"
      ? ["AUTO", ...data.openvino_devices]
      : ["auto", "cpu", "cuda"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
    const isOpenvino = engineSelect.value === "openvino";
    smallOption.disabled = isOpenvino;
    smallOption.textContent = isOpenvino ? "small (portable engine only)" : "small";
    modelSelect.value = isOpenvino ? "base" : "small";
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();
}

function connectWebSocket() {
  if (ws) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws/live-translation`);
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "result") {
      appendTranscriptLine(message);
    } else if (message.type === "error") {
      setRunStatus(`Error: ${message.message}`, "error");
      setRunning(false);
    } else if (message.type === "stopped") {
      setRunning(false);
      if (!el("run-status").classList.contains("error")) {
        setRunStatus("Idle");
      }
    }
  };
  ws.onclose = () => { ws = null; };
}

function appendTranscriptLine(result) {
  const container = el("transcript");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const line = document.createElement("p");
  line.className = "transcript-line";
  const time = new Date().toLocaleTimeString();
  line.innerHTML = `<span class="transcript-time">${time}</span><span class="transcript-lang">(${(result.detected_language || "auto").toUpperCase()})</span>${escapeHtml(result.text)}`;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function setRunStatus(text, kind) {
  const status = el("run-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setRunning(isRunning) {
  running = isRunning;
  el("ctl-start").disabled = isRunning;
  el("ctl-stop").disabled = !isRunning;
  for (const id of ["ctl-source", "ctl-audio-device", "ctl-engine", "ctl-compute-device", "ctl-model"]) {
    el(id).disabled = isRunning;
  }
  if (isRunning) {
    // Don't claim "Listening..." yet -- the model may still be loading;
    // the status poll (started below) will show the real phase within
    // 1.5s, including the "live" class once it's actually running.
    setRunStatus("Starting...");
    startStatusPoll("live-translation", "live-translation", "run-status");
  } else {
    stopStatusPoll("live-translation");
  }
}

async function startLiveTranslation() {
  setRunStatus("Starting...");
  try {
    await fetchJSON("/api/live-translation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: el("ctl-source").value,
        audio_device: el("ctl-audio-device").value || null,
        engine: el("ctl-engine").value,
        model_size: el("ctl-model").value,
        compute_device: el("ctl-compute-device").value,
      }),
    });
    setRunning(true);
  } catch (err) {
    setRunStatus(`Error: ${err.message}`, "error");
  }
}

async function stopLiveTranslation() {
  el("ctl-stop").disabled = true;
  setRunStatus("Stopping...");
  try {
    await fetchJSON("/api/live-translation/stop", { method: "POST" });
  } catch (err) {
    setRunStatus(`Error: ${err.message}`, "error");
  }
  setRunning(false);
}

function closeLiveTranslation() {
  el("modal-overlay").classList.add("hidden");
  if (running) stopLiveTranslation();
}

// --- Document Q&A demo ---

let docQaIndexed = false;

async function openDocQA() {
  el("docqa-modal-overlay").classList.remove("hidden");
  await populateDocQaDevices();
}

async function populateDocQaDevices() {
  const data = await fetchJSON("/api/doc-qa/devices");

  const engineSelect = el("docqa-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("docqa-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();

  wireSamplePicker("docqa-sample", data.samples, { "docqa-folder": "folder", "docqa-question": "question" });
}

function setDocQaIngestStatus(text, kind) {
  const status = el("docqa-ingest-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setDocQaBusy(busy) {
  el("docqa-ingest").disabled = busy;
  el("docqa-ask").disabled = busy || !docQaIndexed;
  el("docqa-question").disabled = busy || !docQaIndexed;
  // Unlike the question field, the sample picker also fills the folder
  // field -- it needs to be usable *before* ingest to bootstrap a demo
  // from a cold start, so it's only gated on busy, not on docQaIndexed.
  for (const id of ["docqa-folder", "docqa-engine", "docqa-compute-device", "docqa-reindex", "docqa-sample"]) {
    el(id).disabled = busy;
  }
}

async function runDocQaIngest() {
  const folder = el("docqa-folder").value.trim();
  if (!folder) {
    setDocQaIngestStatus("Enter a folder path first.", "error");
    return;
  }

  docQaIndexed = false;
  setDocQaBusy(true);
  setDocQaIngestStatus("Indexing... (first run downloads models)");
  const stopStatus = pollBrickStatus("doc-qa", "docqa-ingest-status");

  try {
    const result = await fetchJSON("/api/doc-qa/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        folder,
        engine: el("docqa-engine").value,
        compute_device: el("docqa-compute-device").value,
        reindex: el("docqa-reindex").checked,
      }),
    });
    docQaIndexed = true;
    setDocQaIngestStatus(`Indexed ${result.chunks} chunk(s) from ${result.folder}`, "live");
  } catch (err) {
    setDocQaIngestStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    setDocQaBusy(false);
  }
}

async function runDocQaAsk() {
  const question = el("docqa-question").value.trim();
  if (!question) return;

  appendDocQaQuestion(question);
  el("docqa-question").value = "";
  setDocQaBusy(true);

  try {
    const answer = await fetchJSON("/api/doc-qa/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    appendDocQaAnswer(answer);
  } catch (err) {
    appendDocQaAnswer({ text: `Error: ${err.message}`, sources: [] });
  } finally {
    setDocQaBusy(false);
  }
}

function appendDocQaQuestion(question) {
  const container = el("docqa-transcript");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const line = document.createElement("p");
  line.className = "transcript-question";
  line.textContent = `Q: ${question}`;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function appendDocQaAnswer(answer) {
  const container = el("docqa-transcript");

  const answerLine = document.createElement("p");
  answerLine.className = "transcript-answer";
  answerLine.textContent = answer.text;
  container.appendChild(answerLine);

  if (answer.sources && answer.sources.length) {
    const sourcesLine = document.createElement("p");
    sourcesLine.className = "transcript-sources";
    sourcesLine.textContent = "Sources: " + answer.sources
      .map((s) => `${s.source} [${s.score.toFixed(2)}]`)
      .join(", ");
    container.appendChild(sourcesLine);
  }

  container.scrollTop = container.scrollHeight;
}

function closeDocQA() {
  el("docqa-modal-overlay").classList.add("hidden");
}

// --- Hardware telemetry ---

const DEMO_NAMES_BY_ID = {
  "live-translation": "Live Speech Translation",
  "doc-qa": "Local Document Q&A",
  "object-detection": "Object Detection Overlay",
  "screen-ocr": "Screen / Image Text Extraction",
  "meeting-notes": "Live Meeting Notes",
  "webcam-effects": "Webcam Background Effects",
  "voice-clone-studio": "Voice Clone Studio",
  "voice-assistant": "Local Voice Assistant",
  "expense-extract": "Expense Report Extractor",
  "smart-recall": "Local Screen Memory",
  "code-review-assist": "Commit & Code Review Assistant",
  "html-creator": "HTML Creator",
};

function matchGaugeKind(device) {
  const d = (device || "").toUpperCase();
  if (d === "CPU") return "cpu";
  // A specific GPU id ("GPU.0", "GPU.1", or bare "GPU" on a single-GPU
  // machine), not a collapsed "gpu" bucket -- so pinning a demo's stage to
  // one physical GPU lights up only that GPU's gauge, not every GPU's.
  if (d.startsWith("GPU")) return d;
  if (d === "NPU") return "npu";
  return null; // e.g. "AUTO" or "cuda" -- device OpenVINO/faster-whisper picks internally, not pinned to one gauge
}

function renderTelemetry(data) {
  // data.active is a list, not a dict keyed by demo id: a demo like
  // expense-extract can have two entries active at once (one per stage,
  // pinned to two different devices), so two different gauges can each
  // carry their own label from the very same demo simultaneously.
  const activeByKind = {};
  for (const info of data.active || []) {
    const kind = matchGaugeKind(info.device);
    if (!kind) continue;
    const baseName = DEMO_NAMES_BY_ID[info.demo_id] || info.demo_id;
    activeByKind[kind] = info.stage_label ? `${baseName} (${info.stage_label})` : baseName;
  }

  // CPU/NPU are singular gauges (data-device); each GPU gets its own gauge
  // (data-gpu-id), built dynamically from however many GPUs this machine
  // reports -- see initGpuGauges(). A gpu id with no matching DOM gauge
  // (e.g. the OpenVINO-unavailable fallback reading, id "GPU") is simply
  // skipped below rather than guessed at.
  const gauges = {
    cpu: { value: data.cpu_percent, name: null, selector: '.telemetry-gauge[data-device="cpu"]' },
    npu: { value: data.npu_percent, name: data.npu_name, selector: '.telemetry-gauge[data-device="npu"]' },
  };
  for (const gpu of data.gpus || []) {
    gauges[gpu.id] = { value: gpu.percent, name: gpu.name, selector: `.telemetry-gauge[data-gpu-id="${gpu.id}"]` };
  }

  for (const [kind, { value, name, selector }] of Object.entries(gauges)) {
    // querySelectorAll, not querySelector: the same gauges appear once in
    // the header and once more per demo modal's footer (cloned from
    // #telemetry-footer-template in initTelemetryFooters), and every copy
    // needs to stay in sync on each poll.
    const gaugeInstances = document.querySelectorAll(selector);
    if (!gaugeInstances.length) continue;
    const activeLabel = activeByKind[kind];

    for (const gauge of gaugeInstances) {
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

      if (activeLabel) {
        noteEl.textContent = activeLabel;
        gauge.classList.add("active-gauge");
      } else {
        noteEl.textContent = name || "";
        gauge.classList.remove("active-gauge");
      }
    }
  }
}

async function pollTelemetry() {
  try {
    renderTelemetry(await fetchJSON("/api/telemetry"));
  } catch {
    // Best-effort panel -- ignore a transient failure and try again next tick.
  }
}

function initTelemetryFooters() {
  const template = el("telemetry-footer-template");
  for (const modal of document.querySelectorAll(".demo-modal")) {
    modal.appendChild(template.content.cloneNode(true));
  }
}

// Every OpenVINO-visible GPU on this machine, fetched once at startup:
// {id, full_name}[]. Drives both the per-GPU telemetry gauges below and the
// friendly labels on every brick's compute-device dropdown (via
// gpuDeviceLabel(), called from each brick's device-select population).
let GPU_DEVICES = [];

async function loadGpuDevices() {
  try {
    GPU_DEVICES = await fetchJSON("/api/system/gpu-devices");
  } catch {
    GPU_DEVICES = [];
  }
}

function gpuDeviceLabel(id) {
  const gpu = GPU_DEVICES.find((g) => g.id === id);
  return gpu ? gpu.full_name : id;
}

// Populates a "try a sample" <select> (samplePickerId) from a brick's
// /devices response (samples: [{name, description, ...payload}]) and wires
// it to fill one or more target fields from the picked sample on change,
// then reset back to the placeholder -- a one-shot insert, not a
// persistent selection. fieldMap is {targetElementId: sampleFieldName};
// a sample field that's null/undefined (e.g. html-creator's document-mode
// sample has no `prompt`) is left alone rather than overwriting the
// target with "null". A <select> target fires a change event (so mode-
// toggle show/hide logic reacts), everything else fires input.
function wireSamplePicker(samplePickerId, samples, fieldMap) {
  const picker = el(samplePickerId);
  if (!picker) return;
  while (picker.options.length > 1) picker.remove(1);
  for (const sample of samples || []) {
    const opt = document.createElement("option");
    opt.value = sample.name;
    opt.textContent = `${sample.name} — ${sample.description}`;
    picker.appendChild(opt);
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
    }
    picker.value = "";
  };
}

// Polls /api/status while a brick call is in flight and reflects its real
// backend phase (loading a model vs. actively running) into a status
// element -- so a slow first-time model download/compile shows *something*
// moving instead of a static "Generating..." the whole time. Returns a
// stop() function; call it once the caller's own fetch settles. Falls back
// to whatever text is already in the element (typically set synchronously
// by the caller before this starts) until the first poll resolves, so this
// can only ever add information, never blank something out.
function pollBrickStatus(demoId, statusElementId) {
  const target = el(statusElementId);
  const intervalId = setInterval(async () => {
    let data;
    try {
      data = await fetchJSON("/api/status");
    } catch {
      return; // best-effort -- a missed poll just skips this tick
    }
    const status = data[demoId];
    if (!target || !status) return;
    const icon = status.phase === "loading" ? "⏳" : status.phase === "error" ? "⚠" : "▶";
    target.textContent = `${icon} ${status.message || status.phase}`;
    target.classList.toggle("error", status.phase === "error");
    target.classList.toggle("live", status.phase === "running");
  }, 1500);
  return () => clearInterval(intervalId);
}

// Named-slot registry on top of pollBrickStatus, for the streaming bricks
// whose "in flight" window is "while running", not "during one fetch" --
// start alongside setXRunning(true), stop alongside setXRunning(false),
// keyed by a short string so each streaming brick doesn't need its own
// module-level stop-function variable.
const _statusPolls = {};
function startStatusPoll(key, demoId, statusElementId) {
  stopStatusPoll(key);
  _statusPolls[key] = pollBrickStatus(demoId, statusElementId);
}
function stopStatusPoll(key) {
  if (_statusPolls[key]) {
    _statusPolls[key]();
    delete _statusPolls[key];
  }
}

// Builds one telemetry gauge per detected GPU inside every
// .telemetry-gpu-gauges placeholder (the header strip plus each modal
// footer, already present after initTelemetryFooters()). GPU_DEVICES is the
// source of truth for *which* gauges exist; /api/telemetry polls only fill
// in their values.
function initGpuGauges() {
  const template = el("telemetry-gpu-gauge-template");
  for (const container of document.querySelectorAll(".telemetry-gpu-gauges")) {
    for (const gpu of GPU_DEVICES) {
      const gauge = template.content.cloneNode(true).querySelector(".telemetry-gauge");
      gauge.dataset.gpuId = gpu.id;
      if (GPU_DEVICES.length > 1) {
        gauge.querySelector(".telemetry-gauge-label").textContent = gpu.id;
      }
      container.appendChild(gauge);
    }
  }
}

async function initTelemetry() {
  initTelemetryFooters();
  await loadGpuDevices();
  initGpuGauges();
  pollTelemetry();
  setInterval(pollTelemetry, 2000);
}

// --- Object detection demo ---

let objdetRunning = false;
let objdetDetectionsTimer = null;

async function openObjectDetection() {
  el("objdet-modal-overlay").classList.remove("hidden");
  await populateObjectDetectionDevices();
}

async function populateObjectDetectionDevices() {
  const data = await fetchJSON("/api/object-detection/devices");

  const sourceSelect = el("objdet-source");
  const sourceDeviceSelect = el("objdet-source-device");
  const fillSourceDevices = () => {
    sourceDeviceSelect.innerHTML = "";
    if (sourceSelect.value === "webcam") {
      if (!data.cameras.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No camera found";
        sourceDeviceSelect.appendChild(opt);
      }
      for (const index of data.cameras) {
        const opt = document.createElement("option");
        opt.value = index;
        opt.textContent = `Camera ${index}`;
        sourceDeviceSelect.appendChild(opt);
      }
    } else {
      for (const screen of data.screens) {
        const opt = document.createElement("option");
        opt.value = screen.index;
        opt.textContent = `Screen ${screen.index} (${screen.width}x${screen.height})`;
        sourceDeviceSelect.appendChild(opt);
      }
    }
  };
  sourceSelect.onchange = fillSourceDevices;
  fillSourceDevices();

  const engineSelect = el("objdet-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (YOLO11n, ${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("objdet-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();
}

function setObjdetStatus(text, kind) {
  const status = el("objdet-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setObjdetRunning(isRunning) {
  objdetRunning = isRunning;
  el("objdet-start").disabled = isRunning;
  el("objdet-stop").disabled = !isRunning;
  for (const id of ["objdet-source", "objdet-source-device", "objdet-engine", "objdet-compute-device"]) {
    el(id).disabled = isRunning;
  }

  const img = el("objdet-video");
  if (isRunning) {
    setObjdetStatus("Running...", "live");
    startStatusPoll("object-detection", "object-detection", "objdet-status");
    img.src = `/api/object-detection/stream?t=${Date.now()}`;
    img.classList.add("visible");
    objdetDetectionsTimer = setInterval(pollObjectDetections, 700);
  } else {
    stopStatusPoll("object-detection");
    setObjdetStatus("Idle");
    img.removeAttribute("src");
    img.classList.remove("visible");
    el("objdet-detections").innerHTML = '<p class="transcript-placeholder">Detected objects will be listed here.</p>';
    if (objdetDetectionsTimer) {
      clearInterval(objdetDetectionsTimer);
      objdetDetectionsTimer = null;
    }
  }
}

async function pollObjectDetections() {
  try {
    const data = await fetchJSON("/api/object-detection/detections");
    if (data.error) {
      setObjdetStatus(`Error: ${data.error}`, "error");
      setObjdetRunning(false);
      return;
    }
    renderObjectDetections(data.detections || []);
  } catch {
    // Best-effort -- a transient failure here shouldn't interrupt the video stream.
  }
}

function renderObjectDetections(detections) {
  const container = el("objdet-detections");
  if (!detections.length) {
    container.innerHTML = '<p class="transcript-placeholder">Nothing detected right now.</p>';
    return;
  }
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  container.innerHTML = sorted
    .map(
      (d) =>
        `<div class="objdet-detection-row"><span class="objdet-detection-label">${escapeHtml(d.label)}</span><span class="objdet-detection-score">${Math.round(d.confidence * 100)}%</span></div>`
    )
    .join("");
}

async function startObjectDetection() {
  setObjdetStatus("Starting...");
  const source = el("objdet-source").value;
  const sourceDeviceValue = el("objdet-source-device").value;
  try {
    await fetchJSON("/api/object-detection/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source,
        camera_index: source === "webcam" ? Number(sourceDeviceValue || 0) : 0,
        screen_index: source === "screen" ? Number(sourceDeviceValue || 1) : 1,
        engine: el("objdet-engine").value,
        compute_device: el("objdet-compute-device").value,
      }),
    });
    setObjdetRunning(true);
  } catch (err) {
    setObjdetStatus(`Error: ${err.message}`, "error");
  }
}

async function stopObjectDetection() {
  el("objdet-stop").disabled = true;
  setObjdetStatus("Stopping...");
  try {
    await fetchJSON("/api/object-detection/stop", { method: "POST" });
  } catch (err) {
    setObjdetStatus(`Error: ${err.message}`, "error");
  }
  setObjdetRunning(false);
}

function closeObjectDetection() {
  el("objdet-modal-overlay").classList.add("hidden");
  if (objdetRunning) stopObjectDetection();
}

// --- Screen / image text extraction demo ---

async function openScreenOcr() {
  el("ocr-modal-overlay").classList.remove("hidden");
  await populateOcrDevices();
}

async function populateOcrDevices() {
  const data = await fetchJSON("/api/screen-ocr/devices");

  const sourceSelect = el("ocr-source");
  const sourceDeviceSelect = el("ocr-source-device");
  const sourceDeviceField = el("ocr-source-device-field");
  const uploadField = el("ocr-upload-field");

  const fillSourceDevices = () => {
    const isUpload = sourceSelect.value === "upload";
    sourceDeviceField.hidden = isUpload;
    uploadField.hidden = !isUpload;
    if (isUpload) return;

    sourceDeviceSelect.innerHTML = "";
    if (sourceSelect.value === "webcam") {
      if (!data.cameras.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No camera found";
        sourceDeviceSelect.appendChild(opt);
      }
      for (const index of data.cameras) {
        const opt = document.createElement("option");
        opt.value = index;
        opt.textContent = `Camera ${index}`;
        sourceDeviceSelect.appendChild(opt);
      }
    } else {
      for (const screen of data.screens) {
        const opt = document.createElement("option");
        opt.value = screen.index;
        opt.textContent = `Screen ${screen.index} (${screen.width}x${screen.height})`;
        sourceDeviceSelect.appendChild(opt);
      }
    }
  };
  sourceSelect.onchange = fillSourceDevices;
  fillSourceDevices();

  const engineSelect = el("ocr-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (vision-language model, ${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("ocr-compute-device");
  const translateCheckbox = el("ocr-translate");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const isOpenvino = engineSelect.value === "openvino";
    const options = isOpenvino ? ["AUTO", ...data.openvino_devices] : ["cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
    translateCheckbox.disabled = !isOpenvino;
    if (!isOpenvino) translateCheckbox.checked = false;
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();
}

function setOcrStatus(text, kind) {
  const status = el("ocr-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function renderOcrResult(data) {
  const container = el("ocr-result");
  container.innerHTML = "";

  const label = document.createElement("p");
  label.className = "ocr-result-label";
  label.textContent = data.translated_text !== null ? "Translation (English)" : "Extracted text";
  container.appendChild(label);

  const textBlock = document.createElement("p");
  textBlock.className = "ocr-text-block";
  const shownText = data.translated_text !== null ? data.translated_text : data.text;
  textBlock.textContent = shownText || "(no text detected)";
  container.appendChild(textBlock);

  if (data.regions && data.regions.length) {
    const regionsLabel = document.createElement("p");
    regionsLabel.className = "ocr-result-label";
    regionsLabel.textContent = "Detected regions";
    container.appendChild(regionsLabel);

    for (const region of data.regions) {
      const row = document.createElement("div");
      row.className = "ocr-region-row";
      row.innerHTML = `<span class="ocr-region-text">${escapeHtml(region.text)}</span><span class="ocr-region-score">${Math.round(region.confidence * 100)}%</span>`;
      container.appendChild(row);
    }
  }
}

async function runScreenOcrExtract() {
  const source = el("ocr-source").value;
  const engine = el("ocr-engine").value;
  const computeDevice = el("ocr-compute-device").value;
  const translate = el("ocr-translate").checked;

  el("ocr-extract").disabled = true;
  setOcrStatus(source === "upload" ? "Uploading and extracting..." : "Capturing and extracting...");
  const stopStatus = pollBrickStatus("screen-ocr", "ocr-status");

  try {
    let data;
    if (source === "upload") {
      const fileInput = el("ocr-upload");
      if (!fileInput.files.length) {
        setOcrStatus("Choose an image file first.", "error");
        el("ocr-extract").disabled = false;
        return;
      }
      const form = new FormData();
      form.append("file", fileInput.files[0]);
      form.append("engine", engine);
      form.append("compute_device", computeDevice);
      form.append("translate", translate);
      data = await fetchJSON("/api/screen-ocr/extract-upload", { method: "POST", body: form });
    } else {
      const sourceDeviceValue = el("ocr-source-device").value;
      data = await fetchJSON("/api/screen-ocr/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source,
          camera_index: source === "webcam" ? Number(sourceDeviceValue || 0) : 0,
          screen_index: source === "screen" ? Number(sourceDeviceValue || 1) : 1,
          engine,
          compute_device: computeDevice,
          translate,
        }),
      });
    }
    renderOcrResult(data);
    setOcrStatus("Done", "live");
  } catch (err) {
    setOcrStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    el("ocr-extract").disabled = false;
  }
}

function closeScreenOcr() {
  el("ocr-modal-overlay").classList.add("hidden");
}

// --- Live meeting notes demo ---
// A stream (live transcript, same WebSocket pattern as live-translation)
// plus a request/response layered on top (Generate notes, same pattern as
// doc-qa's Ask) -- this demo genuinely needs both.

let mtgWs = null;
let mtgRunning = false;

async function openMeetingNotes() {
  el("mtg-modal-overlay").classList.remove("hidden");
  await populateMeetingNotesDevices();
  connectMeetingNotesWebSocket();
}

async function populateMeetingNotesDevices() {
  const data = await fetchJSON("/api/meeting-notes/devices");

  const audioSelect = el("mtg-audio-device");
  const defaultOpt = document.createElement("option");
  defaultOpt.value = "";
  defaultOpt.textContent = "Default device";

  const sourceSelect = el("mtg-source");
  const fillAudioDevices = () => {
    const names = sourceSelect.value === "mic" ? data.microphones : data.speakers;
    audioSelect.innerHTML = "";
    audioSelect.appendChild(defaultOpt.cloneNode(true));
    for (const name of names) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      audioSelect.appendChild(opt);
    }
  };
  sourceSelect.onchange = fillAudioDevices;
  fillAudioDevices();

  const engineSelect = el("mtg-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (Whisper + LLM, ${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("mtg-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["auto", "cpu", "cuda"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();
}

function connectMeetingNotesWebSocket() {
  if (mtgWs) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  mtgWs = new WebSocket(`${protocol}://${location.host}/ws/meeting-notes`);
  mtgWs.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "line") {
      appendMeetingTranscriptLine(message);
    } else if (message.type === "error") {
      setMtgStatus(`Error: ${message.message}`, "error");
      setMtgRunning(false);
    } else if (message.type === "stopped") {
      setMtgRunning(false);
      if (!el("mtg-status").classList.contains("error")) {
        setMtgStatus("Idle");
      }
    }
  };
  mtgWs.onclose = () => { mtgWs = null; };
}

function appendMeetingTranscriptLine(line) {
  const container = el("mtg-transcript");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const p = document.createElement("p");
  p.className = "transcript-line";
  p.innerHTML = `<span class="transcript-time">${line.timestamp}</span><span class="transcript-lang">(${(line.detected_language || "auto").toUpperCase()})</span>${escapeHtml(line.text)}`;
  container.appendChild(p);
  container.scrollTop = container.scrollHeight;
}

function setMtgStatus(text, kind) {
  const status = el("mtg-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setMtgRunning(isRunning) {
  mtgRunning = isRunning;
  el("mtg-start").disabled = isRunning;
  el("mtg-stop").disabled = !isRunning;
  for (const id of ["mtg-source", "mtg-audio-device", "mtg-engine", "mtg-compute-device"]) {
    el(id).disabled = isRunning;
  }
  if (isRunning) {
    // Don't claim "Listening..." yet -- the model may still be loading;
    // the status poll (started below) will show the real phase within
    // 1.5s, including the "live" class once it's actually running.
    setMtgStatus("Starting...");
    startStatusPoll("meeting-notes", "meeting-notes", "mtg-status");
  } else {
    stopStatusPoll("meeting-notes");
  }
}

async function startMeetingNotes() {
  setMtgStatus("Starting...");
  try {
    await fetchJSON("/api/meeting-notes/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: el("mtg-source").value,
        audio_device: el("mtg-audio-device").value || null,
        engine: el("mtg-engine").value,
        compute_device: el("mtg-compute-device").value,
      }),
    });
    setMtgRunning(true);
  } catch (err) {
    setMtgStatus(`Error: ${err.message}`, "error");
  }
}

async function stopMeetingNotes() {
  el("mtg-stop").disabled = true;
  setMtgStatus("Stopping...");
  try {
    await fetchJSON("/api/meeting-notes/stop", { method: "POST" });
  } catch (err) {
    setMtgStatus(`Error: ${err.message}`, "error");
  }
  setMtgRunning(false);
}

function setMtgNotesStatus(text, kind) {
  const status = el("mtg-notes-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function renderMeetingNotes(text) {
  const container = el("mtg-notes");
  container.innerHTML = "";
  const block = document.createElement("p");
  block.className = "ocr-text-block";
  block.textContent = text;
  container.appendChild(block);
}

async function generateMeetingNotes() {
  el("mtg-generate").disabled = true;
  setMtgNotesStatus("Generating...");
  const stopStatus = pollBrickStatus("meeting-notes:notes", "mtg-notes-status");
  try {
    const data = await fetchJSON("/api/meeting-notes/generate", { method: "POST" });
    renderMeetingNotes(data.text);
    setMtgNotesStatus(`Based on ${data.transcript_line_count} transcript line(s)`, "live");
  } catch (err) {
    setMtgNotesStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    el("mtg-generate").disabled = false;
  }
}

function closeMeetingNotes() {
  el("mtg-modal-overlay").classList.add("hidden");
  if (mtgRunning) stopMeetingNotes();
}

// --- Webcam background effects demo ---

let webcamRunning = false;
let webcamStatsTimer = null;

async function openWebcamEffects() {
  el("webcam-modal-overlay").classList.remove("hidden");
  await populateWebcamDevices();
}

async function populateWebcamDevices() {
  const data = await fetchJSON("/api/webcam-effects/devices");

  const cameraSelect = el("webcam-camera");
  cameraSelect.innerHTML = "";
  if (!data.cameras.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "No camera found";
    cameraSelect.appendChild(opt);
  }
  for (const index of data.cameras) {
    const opt = document.createElement("option");
    opt.value = index;
    opt.textContent = `Camera ${index}`;
    cameraSelect.appendChild(opt);
  }

  const engineSelect = el("webcam-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("webcam-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();

  const effectSelect = el("webcam-effect");
  const colorField = el("webcam-color-field");
  effectSelect.onchange = () => {
    colorField.hidden = effectSelect.value !== "replace";
    if (webcamRunning) sendWebcamEffect();
  };
}

async function sendWebcamEffect() {
  try {
    await fetchJSON("/api/webcam-effects/effect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        effect: el("webcam-effect").value,
        color: el("webcam-color").value,
      }),
    });
  } catch {
    // Best-effort -- a failed live effect switch just leaves the previous look on screen.
  }
}

function setWebcamStatus(text, kind) {
  const status = el("webcam-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setWebcamRunning(isRunning) {
  webcamRunning = isRunning;
  el("webcam-start").disabled = isRunning;
  el("webcam-stop").disabled = !isRunning;
  for (const id of ["webcam-camera", "webcam-engine", "webcam-compute-device"]) {
    el(id).disabled = isRunning;
  }

  const img = el("webcam-video");
  if (isRunning) {
    setWebcamStatus("Running...", "live");
    startStatusPoll("webcam-effects", "webcam-effects", "webcam-status");
    img.src = `/api/webcam-effects/stream?t=${Date.now()}`;
    img.classList.add("visible");
    webcamStatsTimer = setInterval(pollWebcamStats, 700);
  } else {
    stopStatusPoll("webcam-effects");
    setWebcamStatus("Idle");
    img.removeAttribute("src");
    img.classList.remove("visible");
    el("webcam-stats").innerHTML = '<p class="transcript-placeholder">Person-coverage will be shown here once running.</p>';
    if (webcamStatsTimer) {
      clearInterval(webcamStatsTimer);
      webcamStatsTimer = null;
    }
  }
}

async function pollWebcamStats() {
  try {
    const data = await fetchJSON("/api/webcam-effects/stats");
    if (data.error) {
      setWebcamStatus(`Error: ${data.error}`, "error");
      setWebcamRunning(false);
      return;
    }
    el("webcam-stats").innerHTML =
      `<div class="objdet-detection-row"><span class="objdet-detection-label">Person coverage</span><span class="objdet-detection-score">${Math.round(data.person_coverage * 100)}%</span></div>`;
  } catch {
    // Best-effort -- a transient failure here shouldn't interrupt the video stream.
  }
}

async function startWebcamEffects() {
  setWebcamStatus("Starting...");
  try {
    await fetchJSON("/api/webcam-effects/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_index: Number(el("webcam-camera").value || 0),
        engine: el("webcam-engine").value,
        compute_device: el("webcam-compute-device").value,
        effect: el("webcam-effect").value,
        color: el("webcam-color").value,
      }),
    });
    setWebcamRunning(true);
  } catch (err) {
    setWebcamStatus(`Error: ${err.message}`, "error");
  }
}

async function stopWebcamEffects() {
  el("webcam-stop").disabled = true;
  setWebcamStatus("Stopping...");
  try {
    await fetchJSON("/api/webcam-effects/stop", { method: "POST" });
  } catch (err) {
    setWebcamStatus(`Error: ${err.message}`, "error");
  }
  setWebcamRunning(false);
}

function closeWebcamEffects() {
  el("webcam-modal-overlay").classList.add("hidden");
  if (webcamRunning) stopWebcamEffects();
}

// --- Expense Report Extractor demo ---

let expxWs = null;
let expxRunning = false;

async function openExpenseExtract() {
  el("expx-modal-overlay").classList.remove("hidden");
  await populateExpenseExtractDevices();
  connectExpenseExtractWebSocket();
}

async function populateExpenseExtractDevices() {
  const data = await fetchJSON("/api/expense-extract/devices");
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;

  for (const engineId of ["expx-ocr-engine", "expx-llm-engine"]) {
    const engineSelect = el(engineId);
    const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
    openvinoOption.disabled = !hasOpenvino;
    openvinoOption.textContent = hasOpenvino
      ? `OpenVINO (${data.openvino_devices.join(", ")})`
      : "OpenVINO (install this brick's `openvino` extra to enable)";
    if (hasOpenvino) engineSelect.value = "openvino";
  }

  const wireComputeDevices = (engineSelectId, deviceSelectId) => {
    const engineSelect = el(engineSelectId);
    const deviceSelect = el(deviceSelectId);
    const fill = () => {
      deviceSelect.innerHTML = "";
      const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
      for (const value of options) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
        deviceSelect.appendChild(opt);
      }
    };
    engineSelect.onchange = fill;
    fill();
  };
  wireComputeDevices("expx-ocr-engine", "expx-ocr-device");
  wireComputeDevices("expx-llm-engine", "expx-llm-device");

  // Nudge the demo toward its actual point: OCR on the GPU, LLM structuring
  // on the NPU, running at once -- if this machine has both. Deliberately
  // GPU-for-OCR / NPU-for-LLM, not the other way around: screen-ocr's
  // OpenVINO engine is a large (7B) vision-language model, and testing
  // found its NPU compile reliably fails on this hardware ("Can't convert
  // 76 Bit to Byte" in OpenVINO's vpux-compiler) -- GPU is the OCR device
  // that's actually verified working. doc-qa's LLM, much smaller, compiles
  // and runs fine on NPU. See expense-extract's README for the finding.
  const anyGpu = data.openvino_devices.find((d) => d.toUpperCase().startsWith("GPU"));
  if (hasOpenvino && anyGpu) {
    el("expx-ocr-engine").value = "openvino";
    el("expx-ocr-engine").dispatchEvent(new Event("change"));
    el("expx-ocr-device").value = anyGpu;
  }
  if (hasOpenvino && data.openvino_devices.includes("NPU")) {
    el("expx-llm-engine").value = "openvino";
    el("expx-llm-engine").dispatchEvent(new Event("change"));
    el("expx-llm-device").value = "NPU";
  }
}

function connectExpenseExtractWebSocket() {
  if (expxWs) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  expxWs = new WebSocket(`${protocol}://${location.host}/ws/expense-extract`);
  expxWs.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "ocr_progress") {
      setExpxStatus(`OCR ${message.index}/${message.total}: ${message.file}`, "live");
    } else if (message.type === "structured") {
      appendExpenseLine(message.line);
    } else if (message.type === "done") {
      setExpxStatus(`Done -- ${message.structured}/${message.count} structured, total $${message.total.toFixed(2)}`, "live");
      setExpxRunning(false);
    } else if (message.type === "error") {
      setExpxStatus(`Error: ${message.message}`, "error");
      setExpxRunning(false);
    } else if (message.type === "stopped") {
      setExpxRunning(false);
      if (!el("expx-status").classList.contains("error")) {
        setExpxStatus("Idle");
      }
    }
  };
  expxWs.onclose = () => { expxWs = null; };
}

function appendExpenseLine(line) {
  const container = el("expx-transcript");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const row = document.createElement("p");
  row.className = "transcript-answer";
  if (line.error) {
    row.textContent = `${line.source_file}: skipped (${line.error})`;
  } else {
    const amount = line.amount !== null && line.amount !== undefined ? `$${line.amount.toFixed(2)}` : "?";
    row.textContent = `${line.source_file}: ${line.vendor || "?"} -- ${line.date || "?"} -- ${amount} -- ${line.category}`;
  }
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function setExpxStatus(text, kind) {
  const status = el("expx-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setExpxRunning(isRunning) {
  expxRunning = isRunning;
  el("expx-start").disabled = isRunning;
  el("expx-stop").disabled = !isRunning;
  for (const id of ["expx-folder", "expx-ocr-engine", "expx-ocr-device", "expx-llm-engine", "expx-llm-device"]) {
    el(id).disabled = isRunning;
  }
  // Two stages run concurrently (OCR + LLM structuring); the single status
  // span shows the OCR stage's phase -- both start together, so it's
  // representative of "is the pipeline actually going" either way.
  if (isRunning) {
    startStatusPoll("expense-extract", "expense-extract:ocr", "expx-status");
  } else {
    stopStatusPoll("expense-extract");
  }
}

async function startExpenseExtract() {
  const folder = el("expx-folder").value.trim();
  if (!folder) {
    setExpxStatus("Enter a folder path first.", "error");
    return;
  }

  el("expx-transcript").innerHTML = '<p class="transcript-placeholder">Results will appear here as each receipt is structured.</p>';
  setExpxStatus("Starting...");
  try {
    await fetchJSON("/api/expense-extract/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        folder,
        ocr_engine: el("expx-ocr-engine").value,
        ocr_compute_device: el("expx-ocr-device").value,
        llm_engine: el("expx-llm-engine").value,
        llm_compute_device: el("expx-llm-device").value,
      }),
    });
    setExpxRunning(true);
  } catch (err) {
    setExpxStatus(`Error: ${err.message}`, "error");
  }
}

async function stopExpenseExtract() {
  el("expx-stop").disabled = true;
  setExpxStatus("Stopping...");
  try {
    await fetchJSON("/api/expense-extract/stop", { method: "POST" });
  } catch (err) {
    setExpxStatus(`Error: ${err.message}`, "error");
  }
  setExpxRunning(false);
}

function closeExpenseExtract() {
  el("expx-modal-overlay").classList.add("hidden");
  if (expxRunning) stopExpenseExtract();
}

// --- Local Screen Memory demo ---

let recallWs = null;
let recallRunning = false;

async function openRecall() {
  el("recall-modal-overlay").classList.remove("hidden");
  await populateRecallDevices();
  await refreshRecallStatus();
  connectRecallWebSocket();
}

async function populateRecallDevices() {
  const data = await fetchJSON("/api/smart-recall/devices");

  const screenSelect = el("recall-screen");
  screenSelect.innerHTML = "";
  for (const screen of data.screens) {
    const opt = document.createElement("option");
    opt.value = screen.index;
    opt.textContent = `Screen ${screen.index} (${screen.width}x${screen.height})`;
    screenSelect.appendChild(opt);
  }

  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  for (const engineId of ["recall-ocr-engine", "recall-embed-engine"]) {
    const openvinoOption = el(engineId).querySelector('option[value="openvino"]');
    openvinoOption.disabled = !hasOpenvino;
    openvinoOption.textContent = hasOpenvino
      ? `OpenVINO (${data.openvino_devices.join(", ")})`
      : "OpenVINO (install this brick's `openvino` extra to enable)";
    if (hasOpenvino) el(engineId).value = "openvino";
  }

  const wireComputeDevices = (engineSelectId, deviceSelectId) => {
    const engineSelect = el(engineSelectId);
    const deviceSelect = el(deviceSelectId);
    const fill = () => {
      deviceSelect.innerHTML = "";
      const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
      for (const value of options) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
        deviceSelect.appendChild(opt);
      }
    };
    engineSelect.onchange = fill;
    fill();
  };
  wireComputeDevices("recall-ocr-engine", "recall-ocr-device");
  wireComputeDevices("recall-embed-engine", "recall-embed-device");

  wireSamplePicker("recall-sample", data.samples, { "recall-question": "question" });
}

async function refreshRecallStatus() {
  const status = await fetchJSON("/api/smart-recall/status");
  const embedEngineSelect = el("recall-embed-engine");

  // Reopening the modal (or a page reload) shouldn't lose track of a
  // recording that's still going server-side -- the background thread
  // outlives any one browser tab, so the UI has to ask rather than assume.
  setRecallRunning(status.running);

  if (status.embed_engine) {
    // The index already has a fixed embedding engine -- lock the selector
    // to it instead of letting the user pick something that would just
    // get rejected (or worse, silently produce meaningless results).
    embedEngineSelect.value = status.embed_engine;
    embedEngineSelect.dispatchEvent(new Event("change"));
    embedEngineSelect.disabled = true;
  } else {
    embedEngineSelect.disabled = status.running;
  }

  if (!status.running) {
    setRecallStatus(status.indexed_count > 0 ? `Idle -- ${status.indexed_count} screen(s) indexed` : "Idle");
  }
}

function connectRecallWebSocket() {
  if (recallWs) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  recallWs = new WebSocket(`${protocol}://${location.host}/ws/smart-recall`);
  recallWs.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "indexed") {
      appendRecallEvent(`[${message.timestamp}] indexed: ${message.chunk.text.slice(0, 100)}`, "transcript-answer");
    } else if (message.type === "skipped") {
      appendRecallEvent(`(skipped -- ${message.reason})`, "transcript-sources");
    } else if (message.type === "error") {
      setRecallStatus(`Error: ${message.message}`, "error");
      setRecallRunning(false);
    } else if (message.type === "stopped") {
      setRecallRunning(false);
      if (!el("recall-status").classList.contains("error")) {
        refreshRecallStatus();
      }
    }
  };
  recallWs.onclose = () => {
    recallWs = null;
    // Reconnect while the modal's still open -- recording is a background
    // thread that outlives any one WebSocket, so a dropped connection
    // (idle timeout, a network blip) shouldn't silently stop the capture
    // feed from updating while the operator is still watching it.
    if (!el("recall-modal-overlay").classList.contains("hidden")) {
      setTimeout(connectRecallWebSocket, 1000);
    }
  };
}

function appendRecallEvent(text, className) {
  const container = el("recall-capture-feed");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const line = document.createElement("p");
  line.className = className;
  line.textContent = text;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function setRecallStatus(text, kind) {
  const status = el("recall-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setRecallRunning(isRunning) {
  recallRunning = isRunning;
  el("recall-start").disabled = isRunning;
  el("recall-stop").disabled = !isRunning;
  el("recall-reset").disabled = isRunning;
  for (const id of ["recall-screen", "recall-interval", "recall-ocr-engine", "recall-ocr-device", "recall-embed-device"]) {
    el(id).disabled = isRunning;
  }
  if (isRunning) {
    setRecallStatus("Recording...", "live");
    startStatusPoll("smart-recall", "smart-recall:ocr", "recall-status");
  } else {
    stopStatusPoll("smart-recall");
  }
}

async function startRecall() {
  setRecallStatus("Starting...");
  try {
    await fetchJSON("/api/smart-recall/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        screen_index: Number(el("recall-screen").value),
        interval_seconds: Number(el("recall-interval").value) || 5,
        ocr_engine: el("recall-ocr-engine").value,
        ocr_compute_device: el("recall-ocr-device").value,
        embed_engine: el("recall-embed-engine").value,
        embed_compute_device: el("recall-embed-device").value,
      }),
    });
    setRecallRunning(true);
  } catch (err) {
    setRecallStatus(`Error: ${err.message}`, "error");
  }
}

async function stopRecall() {
  el("recall-stop").disabled = true;
  setRecallStatus("Stopping...");
  try {
    await fetchJSON("/api/smart-recall/stop", { method: "POST" });
  } catch (err) {
    setRecallStatus(`Error: ${err.message}`, "error");
  }
  setRecallRunning(false);
}

async function resetRecall() {
  if (!confirm("Delete every indexed screen capture and screenshot? This can't be undone.")) return;
  try {
    await fetchJSON("/api/smart-recall/reset", { method: "POST" });
    el("recall-embed-engine").disabled = false;
    el("recall-capture-feed").innerHTML = '<p class="transcript-placeholder">Capture events will appear here while recording.</p>';
    el("recall-results").innerHTML = '<p class="transcript-placeholder">Search results, with a screenshot thumbnail, will appear here.</p>';
    await refreshRecallStatus();
  } catch (err) {
    setRecallStatus(`Error: ${err.message}`, "error");
  }
}

function renderRecallResults(results) {
  const container = el("recall-results");
  container.innerHTML = "";

  if (!results.length) {
    container.innerHTML = '<p class="transcript-placeholder">No matches yet.</p>';
    return;
  }

  for (const r of results) {
    const row = document.createElement("div");
    row.className = "recall-result";
    row.innerHTML = `
      <img class="recall-result-thumb" src="${r.screenshot_url}" alt="Screenshot from ${escapeHtml(r.source)}" />
      <div class="recall-result-body">
        <div class="recall-result-meta">
          <span>${escapeHtml(r.source)}</span>
          <span class="recall-result-score">${r.score.toFixed(2)}</span>
        </div>
        <p class="recall-result-text">${escapeHtml(r.text.slice(0, 220))}</p>
      </div>
    `;
    container.appendChild(row);
  }
}

async function runRecallSearch() {
  const question = el("recall-question").value.trim();
  if (!question) return;

  el("recall-search").disabled = true;
  try {
    const data = await fetchJSON("/api/smart-recall/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });
    renderRecallResults(data.results);
  } catch (err) {
    el("recall-results").innerHTML = `<p class="transcript-placeholder">Error: ${escapeHtml(err.message)}</p>`;
  } finally {
    el("recall-search").disabled = false;
  }
}

function closeRecall() {
  el("recall-modal-overlay").classList.add("hidden");
  if (recallRunning) stopRecall();
}

// --- Local Voice Assistant demo ---

let vaWs = null;
let vaRunning = false;

async function openVoiceAssistant() {
  el("va-modal-overlay").classList.remove("hidden");
  await populateVoiceAssistantDevices();
  connectVoiceAssistantWebSocket();
}

async function populateVoiceAssistantDevices() {
  const data = await fetchJSON("/api/voice-assistant/devices");

  const audioSelect = el("va-audio-device");
  audioSelect.innerHTML = "";
  const defaultOpt = document.createElement("option");
  defaultOpt.value = "";
  defaultOpt.textContent = "Default microphone";
  audioSelect.appendChild(defaultOpt);
  for (const name of data.microphones) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    audioSelect.appendChild(opt);
  }

  const wakeSelect = el("va-wake-word");
  wakeSelect.innerHTML = "";
  for (const word of data.wake_words) {
    const opt = document.createElement("option");
    opt.value = word;
    opt.textContent = word.replace(/_/g, " ");
    wakeSelect.appendChild(opt);
  }

  const engineSelect = el("va-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("va-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["auto", "cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();
}

function connectVoiceAssistantWebSocket() {
  if (vaWs) return;
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  vaWs = new WebSocket(`${protocol}://${location.host}/ws/voice-assistant`);
  vaWs.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "wake") {
      appendVaNote("Wake word heard -- listening for your question...");
    } else if (message.type === "heard") {
      appendVaLine("transcript-question", `You: ${message.text}`);
    } else if (message.type === "reply") {
      appendVaLine("transcript-answer", `Assistant: ${message.text}`);
    } else if (message.type === "error") {
      setVaStatus(`Error: ${message.message}`, "error");
      setVaRunning(false);
    } else if (message.type === "stopped") {
      setVaRunning(false);
      if (!el("va-status").classList.contains("error")) {
        setVaStatus("Idle");
      }
    }
  };
  vaWs.onclose = () => { vaWs = null; };
}

function appendVaLine(className, text) {
  const container = el("va-transcript");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const line = document.createElement("p");
  line.className = className;
  line.textContent = text;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function appendVaNote(text) {
  const container = el("va-transcript");
  const placeholder = container.querySelector(".transcript-placeholder");
  if (placeholder) placeholder.remove();

  const line = document.createElement("p");
  line.className = "transcript-sources";
  line.textContent = text;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

function setVaStatus(text, kind) {
  const status = el("va-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setVaRunning(isRunning) {
  vaRunning = isRunning;
  el("va-start").disabled = isRunning;
  el("va-stop").disabled = !isRunning;
  for (const id of ["va-audio-device", "va-wake-word", "va-engine", "va-compute-device", "va-speak"]) {
    el(id).disabled = isRunning;
  }
  if (isRunning) {
    setVaStatus("Listening...", "live");
    startStatusPoll("voice-assistant", "voice-assistant", "va-status");
  } else {
    stopStatusPoll("voice-assistant");
  }
}

async function startVoiceAssistant() {
  setVaStatus("Starting...");
  try {
    await fetchJSON("/api/voice-assistant/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_device: el("va-audio-device").value || null,
        engine: el("va-engine").value,
        compute_device: el("va-compute-device").value,
        wake_word: el("va-wake-word").value,
        speak_replies: el("va-speak").checked,
      }),
    });
    setVaRunning(true);
  } catch (err) {
    setVaStatus(`Error: ${err.message}`, "error");
  }
}

async function stopVoiceAssistant() {
  el("va-stop").disabled = true;
  setVaStatus("Stopping...");
  try {
    await fetchJSON("/api/voice-assistant/stop", { method: "POST" });
  } catch (err) {
    setVaStatus(`Error: ${err.message}`, "error");
  }
  setVaRunning(false);
}

function closeVoiceAssistant() {
  el("va-modal-overlay").classList.add("hidden");
  if (vaRunning) stopVoiceAssistant();
}

// --- Voice Clone Studio demo ---

async function openVoiceCloneStudio() {
  el("voice-modal-overlay").classList.remove("hidden");
  await populateVoiceDevices();
  const status = await fetchJSON("/api/voice-clone-studio/status");
  setVoiceEnrolled(status.enrolled);
}

async function populateVoiceDevices() {
  const data = await fetchJSON("/api/voice-clone-studio/devices");

  const sourceSelect = el("voice-source");
  const recordField = el("voice-record-field");
  const uploadField = el("voice-upload-field");
  const fillSource = () => {
    const isUpload = sourceSelect.value === "upload";
    recordField.hidden = isUpload;
    uploadField.hidden = !isUpload;
  };
  sourceSelect.onchange = fillSource;
  fillSource();

  const engineSelect = el("voice-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("voice-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["CPU"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();

  if (!data.microphones || !data.microphones.length) {
    const recordOption = sourceSelect.querySelector('option[value="record"]');
    recordOption.disabled = true;
    recordOption.textContent = "Record from microphone (none found)";
    sourceSelect.value = "upload";
    fillSource();
  }

  wireSamplePicker("voice-sample", data.samples, { "voice-text": "text" });
}

function setVoiceEnrollStatus(text, kind) {
  const status = el("voice-enroll-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function setVoiceEnrolled(enrolled) {
  el("voice-text").disabled = !enrolled;
  el("voice-sample").disabled = !enrolled;
  el("voice-style").disabled = !enrolled;
  el("voice-tau").disabled = !enrolled;
  el("voice-synthesize").disabled = !enrolled;
  if (enrolled) setVoiceEnrollStatus("Voice enrolled -- ready to speak", "live");
}

async function runVoiceEnroll() {
  const source = el("voice-source").value;
  const engine = el("voice-engine").value;
  const computeDevice = el("voice-compute-device").value;

  el("voice-enroll").disabled = true;
  setVoiceEnrollStatus(source === "record" ? "Recording..." : "Uploading and enrolling...");
  const stopStatus = pollBrickStatus("voice-clone-studio", "voice-enroll-status");

  try {
    if (source === "upload") {
      const fileInput = el("voice-upload");
      if (!fileInput.files.length) {
        setVoiceEnrollStatus("Choose an audio file first.", "error");
        el("voice-enroll").disabled = false;
        return;
      }
      const form = new FormData();
      form.append("file", fileInput.files[0]);
      form.append("engine", engine);
      form.append("compute_device", computeDevice);
      await fetchJSON("/api/voice-clone-studio/enroll-upload", { method: "POST", body: form });
    } else {
      const seconds = Number(el("voice-record-seconds").value) || 10;
      await fetchJSON("/api/voice-clone-studio/enroll-record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds, engine, compute_device: computeDevice }),
      });
    }
    setVoiceEnrolled(true);
  } catch (err) {
    setVoiceEnrollStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    el("voice-enroll").disabled = false;
  }
}

function setVoiceSynthesizeStatus(text, kind) {
  const status = el("voice-synthesize-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

async function runVoiceSynthesize() {
  const text = el("voice-text").value.trim();
  if (!text) {
    setVoiceSynthesizeStatus("Type something to say first.", "error");
    return;
  }

  el("voice-synthesize").disabled = true;
  setVoiceSynthesizeStatus("Synthesizing...");
  const stopStatus = pollBrickStatus("voice-clone-studio", "voice-synthesize-status");

  try {
    const res = await fetch("/api/voice-clone-studio/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, style: el("voice-style").value, tau: Number(el("voice-tau").value) }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `synthesize failed (${res.status})`);
    }
    const blob = await res.blob();
    const player = el("voice-player");
    player.src = URL.createObjectURL(blob);
    player.hidden = false;
    player.play();
    setVoiceSynthesizeStatus("Done", "live");
  } catch (err) {
    setVoiceSynthesizeStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    el("voice-synthesize").disabled = false;
  }
}

function closeVoiceCloneStudio() {
  el("voice-modal-overlay").classList.add("hidden");
}

// --- Init ---

async function loadDeviceSummary() {
  try {
    const data = await fetchJSON("/api/live-translation/devices");
    const ov = data.openvino_devices.length ? data.openvino_devices.join(", ") : "not installed";
    el("device-summary").textContent = `Inference devices: ${ov}\nMicrophones: ${data.microphones.length} · Outputs: ${data.speakers.length}`;
  } catch {
    el("device-summary").textContent = "";
  }
}

async function loadVersion() {
  try {
    const data = await fetchJSON("/api/version");
    el("app-version").textContent = `v${data.version}`;
  } catch {
    // Best-effort -- an empty footer label beats breaking page load over it.
  }
}

// --- Commit & Code Review Assistant demo ---

async function openCodeReviewAssist() {
  el("cra-modal-overlay").classList.remove("hidden");
  await populateCodeReviewDevices();
}

async function populateCodeReviewDevices() {
  const data = await fetchJSON("/api/code-review-assist/devices");

  const sourceSelect = el("cra-source");
  const folderField = el("cra-folder-field");
  const againstField = el("cra-against-field");
  const diffTextField = el("cra-diff-text-field");
  const fillSource = () => {
    const isWorktree = sourceSelect.value === "worktree";
    folderField.hidden = !isWorktree;
    againstField.hidden = !isWorktree;
    diffTextField.hidden = isWorktree;
  };
  sourceSelect.onchange = fillSource;
  fillSource();

  const engineSelect = el("cra-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (coding model, ${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("cra-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
    // This brick's default OpenVINO model is picked to run well on the B60 --
    // prefer GPU.1 when it's present, fall back to AUTO on a machine without it.
    if (engineSelect.value === "openvino" && data.openvino_devices.includes("GPU.1")) {
      computeSelect.value = "GPU.1";
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();

  wireSamplePicker("cra-sample", data.samples, { "cra-diff-text": "diff_text" });
}

function setCraStatus(text, kind) {
  const status = el("cra-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function renderCodeReviewResult(data) {
  const container = el("cra-result");
  container.innerHTML = "";

  if (data.diff_truncated) {
    const warning = document.createElement("p");
    warning.className = "ocr-result-label";
    warning.textContent = `Diff was ${data.diff_char_count} characters -- truncated before review, some changes may not be reflected.`;
    container.appendChild(warning);
  }

  const commitLabel = document.createElement("p");
  commitLabel.className = "ocr-result-label";
  commitLabel.textContent = "Commit message";
  container.appendChild(commitLabel);

  const commitBlock = document.createElement("pre");
  commitBlock.className = "ocr-text-block";
  commitBlock.textContent = data.commit_message;
  container.appendChild(commitBlock);

  const notesLabel = document.createElement("p");
  notesLabel.className = "ocr-result-label";
  notesLabel.textContent = "Review notes";
  container.appendChild(notesLabel);

  const notesBlock = document.createElement("p");
  notesBlock.className = "ocr-text-block";
  notesBlock.textContent = data.review_notes;
  container.appendChild(notesBlock);
}

async function runCodeReview() {
  const source = el("cra-source").value;
  const engine = el("cra-engine").value;
  const computeDevice = el("cra-compute-device").value;

  if (source === "diff_text" && !el("cra-diff-text").value.trim()) {
    setCraStatus("Paste a diff first.", "error");
    return;
  }
  if (source === "worktree" && !el("cra-folder").value.trim()) {
    setCraStatus("Enter a git repo folder first.", "error");
    return;
  }

  el("cra-review").disabled = true;
  setCraStatus("Reviewing...");
  const stopStatus = pollBrickStatus("code-review-assist", "cra-status");

  try {
    const data = await fetchJSON("/api/code-review-assist/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source,
        folder: el("cra-folder").value,
        against: el("cra-against").value || "HEAD",
        diff_text: el("cra-diff-text").value,
        engine,
        compute_device: computeDevice,
      }),
    });
    renderCodeReviewResult(data);
    setCraStatus("Done", "live");
  } catch (err) {
    setCraStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    el("cra-review").disabled = false;
  }
}

function closeCodeReviewAssist() {
  el("cra-modal-overlay").classList.add("hidden");
}

// --- HTML Creator demo ---

let htmlcCurrentHtml = null;

async function openHtmlCreator() {
  el("htmlc-modal-overlay").classList.remove("hidden");
  await populateHtmlCreatorDevices();
}

async function populateHtmlCreatorDevices() {
  const data = await fetchJSON("/api/html-creator/devices");

  const modeSelect = el("htmlc-mode");
  const promptField = el("htmlc-prompt-field");
  const folderField = el("htmlc-folder-field");
  const fillMode = () => {
    const isLandingPage = modeSelect.value === "landing_page";
    promptField.hidden = !isLandingPage;
    folderField.hidden = isLandingPage;
  };
  modeSelect.onchange = fillMode;
  fillMode();

  const engineSelect = el("htmlc-engine");
  const openvinoOption = engineSelect.querySelector('option[value="openvino"]');
  const hasOpenvino = data.openvino_devices && data.openvino_devices.length > 0;
  openvinoOption.disabled = !hasOpenvino;
  openvinoOption.textContent = hasOpenvino
    ? `OpenVINO (coding model, ${data.openvino_devices.join(", ")})`
    : "OpenVINO (install this brick's `openvino` extra to enable)";
  if (hasOpenvino) engineSelect.value = "openvino";

  const computeSelect = el("htmlc-compute-device");
  const fillComputeDevices = () => {
    computeSelect.innerHTML = "";
    const options = engineSelect.value === "openvino" ? ["AUTO", ...data.openvino_devices] : ["cpu"];
    for (const value of options) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = value.toUpperCase().startsWith("GPU") ? gpuDeviceLabel(value) : value;
      computeSelect.appendChild(opt);
    }
    // This brick's default OpenVINO model is picked to run well on the B60 --
    // prefer GPU.1 when it's present, fall back to AUTO on a machine without it.
    if (engineSelect.value === "openvino" && data.openvino_devices.includes("GPU.1")) {
      computeSelect.value = "GPU.1";
    }
  };
  engineSelect.onchange = fillComputeDevices;
  fillComputeDevices();

  wireSamplePicker("htmlc-sample", data.samples, {
    "htmlc-mode": "mode",
    "htmlc-prompt": "prompt",
    "htmlc-folder": "folder",
  });
}

function setHtmlCreatorStatus(text, kind) {
  const status = el("htmlc-status");
  status.textContent = text;
  status.classList.remove("live", "error");
  if (kind) status.classList.add(kind);
}

function renderHtmlCreatorResult(data) {
  htmlcCurrentHtml = data.html;

  const container = el("htmlc-result");
  container.innerHTML = "";

  if (data.html_truncated) {
    const warning = document.createElement("p");
    warning.className = "ocr-result-label";
    warning.textContent = "Output doesn't end with </html> -- it may have been cut off.";
    container.appendChild(warning);
  }
  if (data.source_truncated) {
    const warning = document.createElement("p");
    warning.className = "ocr-result-label";
    warning.textContent = `Source was ${data.source_char_count} characters -- truncated before generation, some content may not be reflected.`;
    container.appendChild(warning);
  }

  const iframe = document.createElement("iframe");
  iframe.className = "htmlc-preview-frame";
  iframe.setAttribute("sandbox", "allow-scripts");
  iframe.srcdoc = data.html;
  container.appendChild(iframe);

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "View raw HTML";
  details.appendChild(summary);
  const pre = document.createElement("pre");
  pre.className = "ocr-text-block";
  pre.textContent = data.html;
  details.appendChild(pre);
  container.appendChild(details);

  el("htmlc-download").hidden = false;
}

function downloadHtmlCreatorResult() {
  if (!htmlcCurrentHtml) return;
  const blob = new Blob([htmlcCurrentHtml], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "generated.html";
  link.click();
  URL.revokeObjectURL(url);
}

async function runHtmlCreator() {
  const mode = el("htmlc-mode").value;
  const engine = el("htmlc-engine").value;
  const computeDevice = el("htmlc-compute-device").value;

  if (mode === "landing_page" && !el("htmlc-prompt").value.trim()) {
    setHtmlCreatorStatus("Describe the page first.", "error");
    return;
  }
  if (mode === "document" && !el("htmlc-folder").value.trim()) {
    setHtmlCreatorStatus("Enter a folder first.", "error");
    return;
  }

  el("htmlc-generate").disabled = true;
  el("htmlc-download").hidden = true;
  setHtmlCreatorStatus("Generating...");
  const stopStatus = pollBrickStatus("html-creator", "htmlc-status");

  try {
    const data = await fetchJSON("/api/html-creator/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        prompt: el("htmlc-prompt").value,
        folder: el("htmlc-folder").value,
        engine,
        compute_device: computeDevice,
      }),
    });
    renderHtmlCreatorResult(data);
    setHtmlCreatorStatus("Done", "live");
  } catch (err) {
    setHtmlCreatorStatus(`Error: ${err.message}`, "error");
  } finally {
    stopStatus();
    el("htmlc-generate").disabled = false;
  }
}

function closeHtmlCreator() {
  el("htmlc-modal-overlay").classList.add("hidden");
}

// --- Activity Log ---

function formatLogTime(atSeconds) {
  return new Date(atSeconds * 1000).toLocaleTimeString();
}

function logDemoLabel(demoId) {
  // demo_id can carry a stage suffix, e.g. "expense-extract:ocr" or
  // "meeting-notes:notes" -- show the friendly brick name plus the stage.
  const [baseId, stage] = demoId.split(":");
  const base = DEMO_NAMES_BY_ID[baseId] || baseId;
  return stage ? `${base} (${stage})` : base;
}

async function openLogViewer() {
  el("log-modal-overlay").classList.remove("hidden");
  const container = el("log-list");
  try {
    const entries = await fetchJSON("/api/logs?limit=100");
    if (!entries.length) {
      container.innerHTML = '<p class="transcript-placeholder">No events yet -- launch a brick to see activity here.</p>';
      return;
    }
    container.innerHTML = "";
    for (const entry of entries.slice().reverse()) {
      const row = document.createElement("div");
      row.className = `log-entry${entry.phase === "error" ? " error" : ""}`;
      row.innerHTML = `<span class="log-entry-time">${formatLogTime(entry.at)}</span><span class="log-entry-demo">${escapeHtml(logDemoLabel(entry.demo_id))}</span><span class="log-entry-message">${escapeHtml(entry.message || entry.phase)}</span>`;
      container.appendChild(row);
    }
  } catch (err) {
    container.innerHTML = `<p class="transcript-placeholder">Error loading log: ${escapeHtml(err.message)}</p>`;
  }
}

function closeLogViewer() {
  el("log-modal-overlay").classList.add("hidden");
}

async function init() {
  const demos = await fetchJSON("/api/demos");
  renderCards(demos);
  loadDeviceSummary();
  loadVersion();
  initTelemetry();

  el("modal-close").addEventListener("click", closeLiveTranslation);
  el("ctl-start").addEventListener("click", startLiveTranslation);
  el("ctl-stop").addEventListener("click", stopLiveTranslation);
  document.querySelector(".placeholder-close").addEventListener("click", () => {
    el("placeholder-overlay").classList.add("hidden");
  });

  el("docqa-modal-close").addEventListener("click", closeDocQA);
  el("docqa-ingest").addEventListener("click", runDocQaIngest);
  el("docqa-ask").addEventListener("click", runDocQaAsk);
  el("docqa-question").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runDocQaAsk();
  });

  el("objdet-modal-close").addEventListener("click", closeObjectDetection);
  el("objdet-start").addEventListener("click", startObjectDetection);
  el("objdet-stop").addEventListener("click", stopObjectDetection);

  el("ocr-modal-close").addEventListener("click", closeScreenOcr);
  el("ocr-extract").addEventListener("click", runScreenOcrExtract);

  el("mtg-modal-close").addEventListener("click", closeMeetingNotes);
  el("mtg-start").addEventListener("click", startMeetingNotes);
  el("mtg-stop").addEventListener("click", stopMeetingNotes);
  el("mtg-generate").addEventListener("click", generateMeetingNotes);

  el("webcam-modal-close").addEventListener("click", closeWebcamEffects);
  el("webcam-start").addEventListener("click", startWebcamEffects);
  el("webcam-stop").addEventListener("click", stopWebcamEffects);
  el("webcam-color").addEventListener("change", () => {
    if (webcamRunning) sendWebcamEffect();
  });

  el("voice-modal-close").addEventListener("click", closeVoiceCloneStudio);
  el("voice-enroll").addEventListener("click", runVoiceEnroll);
  el("voice-synthesize").addEventListener("click", runVoiceSynthesize);

  el("va-modal-close").addEventListener("click", closeVoiceAssistant);
  el("va-start").addEventListener("click", startVoiceAssistant);
  el("va-stop").addEventListener("click", stopVoiceAssistant);

  el("expx-modal-close").addEventListener("click", closeExpenseExtract);
  el("expx-start").addEventListener("click", startExpenseExtract);
  el("expx-stop").addEventListener("click", stopExpenseExtract);

  el("recall-modal-close").addEventListener("click", closeRecall);
  el("recall-start").addEventListener("click", startRecall);
  el("recall-stop").addEventListener("click", stopRecall);
  el("recall-reset").addEventListener("click", resetRecall);
  el("recall-search").addEventListener("click", runRecallSearch);
  el("recall-question").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runRecallSearch();
  });

  el("cra-modal-close").addEventListener("click", closeCodeReviewAssist);
  el("cra-review").addEventListener("click", runCodeReview);

  el("htmlc-modal-close").addEventListener("click", closeHtmlCreator);
  el("htmlc-generate").addEventListener("click", runHtmlCreator);
  el("htmlc-download").addEventListener("click", downloadHtmlCreatorResult);

  el("log-open").addEventListener("click", openLogViewer);
  el("log-modal-close").addEventListener("click", closeLogViewer);
}

init();
