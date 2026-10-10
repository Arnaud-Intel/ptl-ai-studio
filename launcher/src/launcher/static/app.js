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

function appendTimedLine(container, time, lang, text, energy = null) {
  // `energy`, when the runner measured it: what this line cost (BACKLOG R18).
  const cost = energy
    ? `<span class="line-energy" title="${escapeHtml(energyTitle(energy))}">${escapeHtml(energyText(energy))}</span>`
    : "";
  appendLine(
    container,
    "line",
    `<span class="line-time">${escapeHtml(time)}</span><span class="line-lang">(${escapeHtml((lang || "auto").toUpperCase())})</span>${escapeHtml(text)}${cost}`,
    { html: true },
  );
}

// What a result cost: joules above the idle baseline once the launcher has
// learned one, else the processor package's total (BACKLOG R18).
function energyText(energy) {
  const above = energy.above_idle_joules !== null && energy.above_idle_joules !== undefined;
  const joules = above ? energy.above_idle_joules : energy.joules;
  const shown = joules >= 100 ? `${Math.round(joules)} J` : `${joules.toFixed(1)} J`;
  return above ? `+${shown}` : shown;
}

function energyTitle(energy) {
  let title = `Processor package energy over ${energy.seconds.toFixed(1)} s: ${energy.joules.toFixed(1)} J`;
  title +=
    energy.above_idle_joules !== null && energy.above_idle_joules !== undefined
      ? `, ${energy.above_idle_joules.toFixed(1)} J of it above the idle baseline.`
      : " (no idle baseline yet: it needs a few seconds with no demo running).";
  const others = (energy.shared_with || []).map((id) => (demoById(id) || {}).name || id);
  if (others.length) title += ` ${others.join(", ")} ran at the same time, so it isn't this demo's alone.`;
  return title;
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

// ---- Subtitles: what live translation hears, kept in view ---------------------------
//
// For presenting. Turned on, the lines show in a bar docked at the bottom of
// the page whichever demo is open: inside the window being shared in a call,
// in any browser. From the bar they can be popped out into a small window the
// browser keeps above every other application (Document Picture-in-Picture,
// in Chrome and Edge), for subtitles over slides or another program.
const SUBTITLES = { on: false, win: null, panel: null, lines: [], spoken: false, stale: false, note: "", timers: [] };
const SUBTITLE_STALE_MS = 8000; // a line nobody has followed up dims...
const SUBTITLE_CLEAR_MS = 25000; // ...and then leaves the screen
const SUBTITLE_LINES = '<div class="lines"><p class="previous"></p><p class="current"></p><p class="hint"></p></div>';

// The pop-out is its own document: it gets its own copy of the bar's look.
const SUBTITLE_WINDOW_STYLE = `
  html, body { height: 100%; margin: 0; }
  body { background: #05070b; color: #fff; font-family: "Segoe UI", system-ui, sans-serif; overflow: hidden; }
  #stage {
    box-sizing: border-box; height: 100%; padding: 8px 20px; display: flex; flex-direction: column;
    justify-content: center; text-align: center; line-height: 1.25; font-weight: 600;
  }
  .lines { display: flex; flex-direction: column; gap: 0.18em; }
  p { margin: 0; overflow-wrap: anywhere; transition: opacity 0.6s; }
  [hidden] { display: none; }
  .previous { opacity: 0.45; font-weight: 500; }
  .stale .current { opacity: 0.45; }
  .hint { font-size: 14px; font-weight: 400; opacity: 0.6; }
`;

function subtitleWindowsStayOnTop() {
  return "documentPictureInPicture" in window;
}

// Where the lines are drawn now: the pop-out if there is one, else the bar.
function subtitleStage() {
  if (SUBTITLES.win) return SUBTITLES.win.document.getElementById("stage");
  return SUBTITLES.on ? el("subtitle-stage") : null;
}

function clearSubtitleTimers() {
  SUBTITLES.timers.forEach(clearTimeout);
  SUBTITLES.timers = [];
}

function reflectSubtitles() {
  const docked = SUBTITLES.on && !SUBTITLES.win;
  el("lt-subtitles").setAttribute("aria-pressed", String(SUBTITLES.on));
  el("subtitle-bar").hidden = !docked;
  document.body.classList.toggle("subtitles-docked", docked); // room under the page for the bar
  renderSubtitles();
}

function setSubtitles(on, panel = null) {
  if (on === SUBTITLES.on) return;
  SUBTITLES.on = on;
  clearSubtitleTimers();
  if (on) {
    Object.assign(SUBTITLES, { panel, lines: [], spoken: false, stale: false, note: "" });
    // The lines have to keep coming while another demo is on screen, which
    // is the point: the panel keeps its connection until subtitles are
    // turned off (it normally lets go when you leave it).
    panel.stayConnected = true;
    panel.connect();
  } else {
    const { win, panel: was } = SUBTITLES;
    SUBTITLES.win = null;
    SUBTITLES.panel = null;
    if (win) win.close();
    if (was) {
      was.stayConnected = false;
      if (!was.isOpen) was.disconnect(); // nobody is reading the lines any more
    }
  }
  reflectSubtitles();
}

async function popOutSubtitles() {
  if (!SUBTITLES.on || SUBTITLES.win) return;
  let win = null;
  if (subtitleWindowsStayOnTop()) {
    try {
      win = await window.documentPictureInPicture.requestWindow({ width: 920, height: 170 });
    } catch {
      win = null; // an embedded browser can announce the feature and not have it
    }
  }
  if (!win) {
    SUBTITLES.note = "This browser can't keep a window above other applications (Chrome and Edge can). The bar stays here.";
    SUBTITLES.timers.push(setTimeout(() => { SUBTITLES.note = ""; renderSubtitles(); }, 7000));
    renderSubtitles();
    return;
  }
  const doc = win.document;
  doc.title = "Subtitles · Panther Lake AI Studio";
  const style = doc.createElement("style");
  style.textContent = SUBTITLE_WINDOW_STYLE;
  doc.head.append(style);
  doc.body.innerHTML = `<div id="stage">${SUBTITLE_LINES}</div>`;
  SUBTITLES.win = win;
  SUBTITLES.note = "";
  // Closing the window brings the subtitles back to the bar; the switch in
  // the panel is what turns them off.
  win.addEventListener("pagehide", () => {
    if (SUBTITLES.win !== win) return;
    SUBTITLES.win = null;
    reflectSubtitles();
  });
  win.addEventListener("resize", renderSubtitles);
  reflectSubtitles();
}

function showSubtitle(text) {
  if (!SUBTITLES.on || !text) return;
  SUBTITLES.lines = [...SUBTITLES.lines, text].slice(-2);
  SUBTITLES.spoken = true;
  SUBTITLES.stale = false;
  SUBTITLES.note = "";
  clearSubtitleTimers();
  SUBTITLES.timers.push(setTimeout(() => { SUBTITLES.stale = true; renderSubtitles(); }, SUBTITLE_STALE_MS));
  SUBTITLES.timers.push(setTimeout(() => { SUBTITLES.lines = []; renderSubtitles(); }, SUBTITLE_CLEAR_MS));
  renderSubtitles();
}

// The newest line, with the one before it above while both fit, in the
// largest type there is room for.
function renderSubtitles() {
  const stage = subtitleStage();
  if (!stage) return;
  const current = SUBTITLES.lines[SUBTITLES.lines.length - 1] || "";
  const previous = SUBTITLES.lines.length > 1 ? SUBTITLES.lines[0] : "";
  const previousNode = stage.querySelector(".previous");
  const hint = stage.querySelector(".hint");
  stage.classList.toggle("stale", SUBTITLES.stale);
  stage.querySelector(".current").textContent = current;
  previousNode.textContent = previous;
  hint.textContent = SUBTITLES.note || (SUBTITLES.spoken ? "" : "What Live Speech Translation hears will appear here.");
  hint.hidden = !hint.textContent;
  // Measured on the lines themselves: text that spills out of the top of a
  // box is not counted as overflow, so the box cannot be asked.
  const lines = stage.querySelector(".lines");
  const padding = getComputedStyle(stage);
  const room = stage.clientHeight - parseFloat(padding.paddingTop) - parseFloat(padding.paddingBottom);
  const largest = Math.round(Math.max(15, Math.min(room * 0.4, stage.clientWidth * 0.05, 46)));
  // The line before stays only while the newest keeps a comfortable size;
  // a long sentence gets the whole space to itself, shrinking as needed.
  const comfortable = Math.max(20, Math.round(largest * 0.7));
  for (const [withPrevious, smallest] of [[Boolean(previous), comfortable], [false, 13]]) {
    previousNode.hidden = !withPrevious;
    for (let size = largest; size >= smallest; size -= 1) {
      stage.style.fontSize = `${size}px`;
      if (lines.offsetHeight <= room + 1) return;
    }
  } // still too long at the smallest type: it is cut, not hidden
}

// The "Spoken language" menu: detect automatically, then the languages the
// launcher offers. The choice is kept when the panel is opened again.
function fillSpokenLanguages(select, data) {
  const chosen = select.value || "auto";
  const options = [`<option value="auto">Detect automatically</option>`];
  for (const language of data.spoken_languages || []) {
    options.push(`<option value="${escapeHtml(language.code)}">${escapeHtml(language.name)}</option>`);
  }
  select.innerHTML = options.join("");
  select.value = chosen;
  if (select.value !== chosen) select.value = "auto";
}

function deviceLabel(id) {
  const upper = String(id).toUpperCase();
  if (upper.startsWith("GPU")) {
    const gpu = GPU_DEVICES.find((g) => g.id === id);
    return gpu ? gpu.full_name : id;
  }
  if (upper === "AUTO") return "Auto (the app picks the chip)";
  if (upper === "CPU") return "CPU";
  if (upper === "NPU") return "NPU";
  if (upper === "CUDA") return "CUDA (NVIDIA GPU)";
  return id;
}

// How "Auto" names the chip it stands for: "the NPU", "the integrated GPU".
function autoDeviceName(id) {
  const upper = String(id).toUpperCase();
  if (upper.startsWith("NPU")) return "the NPU";
  if (upper === "CPU") return "the CPU";
  const gpu = GPU_DEVICES.find((g) => g.id === id);
  if (gpu) return (gpu.full_name || "").includes("dGPU") ? "the discrete GPU" : "the integrated GPU";
  return deviceLabel(id);
}

// The OpenVINO device to default a *large* model to (a 30B coding LLM, a 7B
// vision model): the discrete GPU if the machine has one among the brick's
// offered devices, since it's faster; otherwise the integrated GPU, which
// holds these models in shared memory; AUTO only with no GPU at all. Mirrors
// pantherlake_ai_core.engine.preferred_large_model_device().
function preferredLargeModelDevice(openvinoDevices) {
  const offered = GPU_DEVICES.filter((g) => (openvinoDevices || []).includes(g.id));
  const discrete = offered.filter((g) => (g.full_name || "").includes("dGPU"));
  if (discrete.length) return discrete[discrete.length - 1].id;
  return offered.length ? offered[0].id : "AUTO";
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

// One line under a generated answer: how fast it came and on which chip --
// the number the audience is meant to see (BACKLOG R20). Empty without stats.
function generationStatsHtml(stats) {
  if (!stats) return "";
  const where = shortGpuName(deviceLabel(stats.device));
  const parts = [
    `<strong>${stats.tokens_per_second.toFixed(1)} tokens/s</strong> on ${escapeHtml(where)}`,
    `${stats.tokens} tokens in ${stats.seconds.toFixed(1)} s`,
  ];
  if (stats.first_token_seconds !== null && stats.first_token_seconds !== undefined) {
    parts.push(`first token after ${stats.first_token_seconds.toFixed(2)} s`);
  }
  if (stats.energy) {
    const above = stats.energy.above_idle_joules !== null && stats.energy.above_idle_joules !== undefined;
    parts.push(
      `<span class="energy" title="${escapeHtml(energyTitle(stats.energy))}">${escapeHtml(energyText(stats.energy))}` +
        `${above ? " above idle" : " (processor package)"}</span>`,
    );
  }
  return `<p class="gen-stats">${parts.join(" · ")}</p>`;
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
// The Model menu of a brick that answers with a small language model
// (data.language_models, from its /devices route). Its first entry leaves the
// choice to the brick and says which model that is on the chip chosen: a
// model comes in one build for the NPU and another for the rest, and a brick
// that prefers the larger one takes it only where this laptop has it -- a
// better answer is not worth gigabytes fetched in front of an audience.
// `autoDevice()` is the chip the brick's "Auto" stands for, when it is known.
function wireLanguageModel(select, data, { engine = null, device, autoDevice = () => null }) {
  const choice = data.language_models;
  const field = select.closest(".field");
  if (field) field.hidden = !choice;
  if (!choice) return;
  const names = Object.fromEntries(choice.models.map((model) => [model.key, model.name]));
  const fill = () => {
    const kept = select.value;
    if (engine && engine.value !== "openvino") {
      // The portable engine has the one model.
      fillSelect(select, [{ value: "", label: names[choice.portable] || choice.portable }]);
      select.dataset.auto = "";
      return;
    }
    const chip = !device.value || device.value === "AUTO" ? autoDevice() || "" : device.value;
    const kind = String(chip).toUpperCase().startsWith("NPU") ? "npu" : "other";
    select.dataset.auto = choice.default[kind];
    fillSelect(select, [
      { value: "", label: `Auto (${names[choice.default[kind]]})` },
      ...choice.models.map((model) => ({
        value: model.key,
        label: `${model.name} -- ${model.says}` + (model.on_disk[kind] ? "" : " (not on this laptop yet: downloaded first)"),
      })),
    ]);
    if ([...select.options].some((o) => o.value === kept)) select.value = kept;
  };
  if (engine) engine.addEventListener("change", fill);
  device.addEventListener("change", fill);
  fill();
}

// The model a Model menu stands for now, by name, for a request: what was
// picked, or what "Auto" means on the chip chosen. null leaves it to the brick.
function languageModelOf(select) {
  return select.value || select.dataset.auto || null;
}

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
    // A brick whose "Auto" is a known chip (data.auto_device: speech goes to
    // the NPU) says which, rather than "the app picks".
    const autoLabel = data.auto_device ? `Auto (${autoDeviceName(data.auto_device)})` : deviceLabel("AUTO");
    fillSelect(
      deviceSelect,
      values.map((value) => ({ value, label: value === "AUTO" ? autoLabel : deviceLabel(value) })),
    );
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
    (panel?.querySelector(".run-row") || panel?.querySelector(".controls") || picker).after(details);
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

// While a brick writes an answer, show it as it grows. The page asks for
// the text so far a few times a second (see generation.py): a missed ask
// loses nothing, since the next one has more. Returns the function that ends
// the following; whatever the brick renders when it finishes replaces this.
const PARTIAL_POLL_MS = 300;

function followPartial(demoId, { target, stage = "default", onUpdate = null }) {
  const query = stage && stage !== "default" ? `?stage=${encodeURIComponent(stage)}` : "";
  let block = null;
  let ended = false;
  const tick = async () => {
    let data;
    try {
      data = await fetchJSON(`/api/bricks/${demoId}/partial${query}`);
    } catch {
      return; // a missed ask: the next one has everything so far
    }
    if (ended || !data.active || !data.text) return;
    if (!block) {
      target.innerHTML =
        '<div class="partial"><div class="partial-head"><span class="partial-label">Writing…</span>' +
        '<button type="button" class="link-btn partial-stop">Stop</button></div>' +
        '<pre class="partial-text"></pre></div>';
      block = target.querySelector(".partial");
      block.querySelector(".partial-stop").addEventListener("click", async (event) => {
        event.currentTarget.disabled = true;
        block.querySelector(".partial-label").textContent = "Stopping…";
        try {
          await postJSON(`/api/bricks/${demoId}/stop${query}`, {});
        } catch {
          // The request itself reports what went wrong, if anything did.
        }
      });
    }
    const text = block.querySelector(".partial-text");
    text.textContent = data.text.trimStart();
    text.scrollTop = text.scrollHeight;
    if (onUpdate) onUpdate();
  };
  const timer = setInterval(tick, PARTIAL_POLL_MS);
  return () => {
    ended = true;
    clearInterval(timer);
  };
}

// An answer that was stopped part-way stays on screen, and says so: half a
// review that reads like a finished one would be worse than none.
function stoppedNoteHtml(data) {
  return data && data.cancelled ? '<p class="stopped-note">Stopped -- this answer is incomplete.</p>' : "";
}

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
  // `partial` ({ target, stage }) shows the answer in `target` as it is
  // written, until `work` renders the finished one over it.
  async run({ button, statusEl = this.statusEl, key, busy, work, done = "Done", partial = null }) {
    button.disabled = true;
    paintStatus(statusEl, busy, "loading");
    if (key) this.watch(key, statusEl);
    const stopFollowing = partial ? followPartial(this.id, partial) : () => {};
    try {
      const result = await work();
      stopFollowing();
      this.unwatch();
      if (result && result.cancelled) paintStatus(statusEl, "Stopped -- the answer is incomplete");
      else paintStatus(statusEl, typeof done === "function" ? done(result) : done, "live");
      return result;
    } catch (err) {
      this.unwatch();
      paintStatus(statusEl, `Error: ${err.message}`, "error");
      return undefined;
    } finally {
      stopFollowing();
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
    // stayConnected: something outside the panel still shows its lines (the
    // subtitles).
    if (!this.stayConnected) this.disconnect();
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
      if (this.isOpen || this.stayConnected) setTimeout(() => this.connect(), 1000);
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
    counting: card.querySelector(".feed-counting").value,
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
  if (preset.counting) card.querySelector(".feed-counting").value = preset.counting;
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
    controls: ["lt-source", "lt-audio-device", "lt-engine", "lt-compute-device", "lt-model", "lt-language"],
    populate(data) {
      wireAudioSource(el("lt-source"), el("lt-audio-device"), data);
      fillSpokenLanguages(el("lt-language"), data);
      el("lt-summarise").hidden = (demoById("meeting-notes") || {}).status !== "available";
      wireEngineAndDevice(el("lt-engine"), el("lt-compute-device"), data, {
        portableDevices: ["cpu", "cuda"],
        onChange: (isOpenvino) => {
          // "medium" is where the English becomes a translation rather than
          // a gist, and it still runs many times faster than real time on
          // the NPU (see _TRANSLATION_SIZE_DEFAULTS in app.py).
          el("lt-model").value = isOpenvino ? "medium" : "small";
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
        language: el("lt-language").value,
      };
    },
    onMessage(message) {
      if (message.type === "result") {
        // The socket replays what was queued while nobody was connected, which
        // may be lines this page has since drawn from the launcher's transcript.
        if (this.draw(message.transcript, message, message.energy)) showSubtitle(message.text);
      }
    },
    // The launcher keeps the transcript -- across Stop and Start, until it is
    // cleared -- and `drawn` is how far into which transcript this page has
    // got. Ids only grow, so an older one is a line from before a clear.
    drawn: { id: 0, seq: 0 },
    draw(id, line, energy = null) {
      if (id < this.drawn.id || (id === this.drawn.id && line.seq <= this.drawn.seq)) return false;
      if (id !== this.drawn.id) this.startTranscript(id);
      this.drawn.seq = line.seq;
      appendTimedLine(el("lt-transcript"), line.timestamp, line.detected_language, line.text, energy);
      return true;
    },
    startTranscript(id) {
      this.drawn = { id, seq: 0 };
      showPlaceholder(el("lt-transcript"), "Translated speech will appear here once you press Start.");
    },
    // Catch up with the launcher's transcript: after a page reload this is
    // where the lines come back from.
    async syncTranscript() {
      let data;
      try {
        data = await fetchJSON("/api/live-translation/transcript");
      } catch {
        return; // new lines still arrive over the socket
      }
      if (data.id > this.drawn.id) this.startTranscript(data.id);
      for (const line of data.lines) this.draw(data.id, line);
    },
    async rehydrate() {
      await StreamPanel.prototype.rehydrate.call(this);
      await this.syncTranscript();
    },
    // One recording, two uses: Meeting Notes writes its summary from what was
    // heard here, rather than listening to the same meeting a second time.
    async summarise() {
      const status = el("lt-summarise-status");
      await this.syncTranscript();
      if (!this.drawn.seq) {
        paintStatus(status, "Nothing has been transcribed yet: press Start and speak first.", "error");
        return;
      }
      paintStatus(status, "");
      // The Meeting Notes panel makes the hand-over once it is open: the notes
      // are written on the chip chosen there.
      PANELS["meeting-notes"].summariseOnOpen = true;
      location.hash = "#/brick/meeting-notes";
    },
    async clearTranscript() {
      const status = el("lt-summarise-status");
      if (this.drawn.seq && !confirm("Clear the transcript and start a new one? This can't be undone.")) return;
      try {
        const data = await fetchJSON("/api/live-translation/transcript", { method: "DELETE" });
        this.startTranscript(data.id);
        paintStatus(status, "");
      } catch (err) {
        paintStatus(status, `Error: ${err.message}`, "error");
      }
    },
    wireExtra() {
      el("lt-summarise").addEventListener("click", () => this.summarise());
      el("lt-clear").addEventListener("click", () => this.clearTranscript());
      el("lt-subtitles").addEventListener("click", () => setSubtitles(!SUBTITLES.on, this));
      el("subtitle-close").addEventListener("click", () => setSubtitles(false));
      el("subtitle-popout").addEventListener("click", popOutSubtitles);
      el("subtitle-popout").hidden = !subtitleWindowsStayOnTop();
      window.addEventListener("resize", renderSubtitles);
    },
  }),

  "meeting-notes": new StreamPanel({
    id: "meeting-notes",
    prefix: "mtg",
    transport: "ws",
    statusKey: "meeting-notes",
    controls: ["mtg-source", "mtg-audio-device", "mtg-engine", "mtg-compute-device", "mtg-language"],
    populate(data) {
      wireAudioSource(el("mtg-source"), el("mtg-audio-device"), data);
      fillSpokenLanguages(el("mtg-language"), data);
      wireEngineAndDevice(el("mtg-engine"), el("mtg-compute-device"), data, { portableDevices: ["cpu", "cuda"] });
      // The notes have a chip of their own. Left to the app it is the NPU,
      // like the transcription: both models are known to run there.
      const devices = data.openvino_devices || [];
      const auto = data.auto_device ? `Auto (${autoDeviceName(data.auto_device)})` : deviceLabel("AUTO");
      const fillNotesDevice = () => {
        const select = el("mtg-notes-device");
        const chosen = select.value;
        const items =
          el("mtg-engine").value === "openvino"
            ? [{ value: "AUTO", label: auto }, ...devices.map((d) => ({ value: d, label: deviceLabel(d) }))]
            : [{ value: "cpu", label: deviceLabel("cpu") }];
        fillSelect(select, items);
        if (items.some((item) => item.value === chosen)) select.value = chosen;
      };
      el("mtg-engine").addEventListener("change", fillNotesDevice);
      fillNotesDevice();
    },
    body() {
      return {
        source: el("mtg-source").value,
        audio_device: el("mtg-audio-device").value || null,
        engine: el("mtg-engine").value,
        compute_device: el("mtg-compute-device").value,
        notes_device: el("mtg-notes-device").value,
        language: el("mtg-language").value,
      };
    },
    onMessage(message) {
      if (message.type === "line") {
        appendTimedLine(el("mtg-transcript"), message.timestamp, message.detected_language, message.text);
      }
    },
    // Following live translation: asked for a summary from that panel, this
    // one shows its transcript as it grows and summarises all of it on
    // demand. Its own Start and Stop step aside meanwhile -- they start a
    // second, separate transcription, which next to a copy of the live
    // transcript read as "restart the live one" and did something else
    // (2026-10-07: a frozen copy, then "Transcribing..." and no lines).
    following: null, // { id, seq }: how far into which live transcript this panel has drawn
    followTimer: null,
    setFollowing(on) {
      el("mtg-panel").classList.toggle("following", on);
      el("mtg-following").hidden = !on;
      clearInterval(this.followTimer);
      this.followTimer = null;
      if (on) {
        this.following = { id: 0, seq: 0 };
        showPlaceholder(el("mtg-transcript"), "Nothing has been said yet.");
        this.followTimer = setInterval(() => this.syncFollowed(), 1500);
      } else {
        this.following = null;
        showPlaceholder(el("mtg-transcript"), "The live transcript will appear here once you press Start.");
      }
    },
    async syncFollowed() {
      if (!this.following || !this.isOpen) return;
      let data;
      try {
        data = await fetchJSON("/api/live-translation/transcript");
      } catch {
        return; // the next tick asks again
      }
      if (!this.following) return;
      if (data.id !== this.following.id) {
        this.following = { id: data.id, seq: 0 }; // cleared over there: start again here too
        showPlaceholder(el("mtg-transcript"), "Nothing has been said yet.");
      }
      for (const line of data.lines) {
        if (line.seq <= this.following.seq) continue;
        this.following.seq = line.seq;
        appendTimedLine(el("mtg-transcript"), line.timestamp, line.detected_language, line.text);
      }
      const live = STATUS.snapshot["live-translation"];
      const name = (demoById("live-translation") || {}).name || "live translation";
      const state = live ? live.message || live.phase : "stopped";
      paintStatus(
        el("mtg-following-status"),
        `Following ${name} (${state}) -- ${data.lines.length} line(s) so far`,
        live ? PHASE_KIND[live.phase] || "loading" : null,
      );
    },
    leave() {
      StreamPanel.prototype.leave.call(this);
      clearInterval(this.followTimer);
      this.followTimer = null;
    },
    async rehydrate() {
      await StreamPanel.prototype.rehydrate.call(this);
      if (this.summariseOnOpen) {
        this.summariseOnOpen = false;
        this.summariseLiveTranslation(); // not awaited: opening the panel doesn't wait for the notes
      } else if (this.following) {
        // Back on a panel that was following: pick the transcript up again.
        this.followTimer = setInterval(() => this.syncFollowed(), 1500);
        this.syncFollowed();
      }
    },
    // Asked for from the live translation panel: follow its transcript and
    // write the notes from what has been said so far.
    async summariseLiveTranslation() {
      if (el("mtg-generate").disabled) return; // notes are being written already
      if (this.running) {
        paintStatus(
          el("mtg-notes-status"),
          "Meeting Notes is transcribing a meeting of its own: stop it before summarising live translation.",
          "error",
        );
        return;
      }
      if (!this.following) this.setFollowing(true);
      await this.syncFollowed();
      await this.generateNotes();
    },
    generateNotes() {
      return this.run({
        button: el("mtg-generate"),
        statusEl: el("mtg-notes-status"),
        key: "meeting-notes:notes",
        partial: { target: el("mtg-notes"), stage: "notes" },
        busy: "Generating notes…",
        work: async () => {
          if (this.following) {
            // Everything live translation has heard up to now, not what it
            // had heard at the last summary.
            await postJSON("/api/meeting-notes/from-live-translation", {
              engine: el("mtg-engine").value,
              notes_device: el("mtg-notes-device").value,
            });
          }
          // The chip chosen now, which may not be the one the last notes were written on.
          const data = await postJSON("/api/meeting-notes/generate", { notes_device: el("mtg-notes-device").value });
          renderTextBlock(el("mtg-notes"), data.text);
          el("mtg-notes").insertAdjacentHTML("afterbegin", stoppedNoteHtml(data));
          return data;
        },
        done: (data) =>
          `Based on ${data.transcript_line_count} transcript line(s)` +
          (data.device ? `, written on ${deviceLabel(data.device)}` : "") +
          (data.parts > 1 ? ` -- a long meeting, summarised in ${data.parts} parts and merged` : ""),
      });
    },
    wireExtra() {
      el("mtg-generate").addEventListener("click", () => this.generateNotes());
      el("mtg-following-open").addEventListener("click", () => (location.hash = "#/brick/live-translation"));
      el("mtg-following-stop").addEventListener("click", () => this.setFollowing(false));
    },
  }),

  "voice-assistant": new StreamPanel({
    id: "voice-assistant",
    prefix: "va",
    transport: "ws",
    statusKey: "voice-assistant",
    controls: ["va-audio-device", "va-wake-word", "va-engine", "va-compute-device", "va-model", "va-speak"],
    populate(data) {
      fillSelect(el("va-audio-device"), [
        { value: "", label: "Default microphone" },
        ...(data.microphones || []).map((n) => ({ value: n, label: n })),
      ]);
      fillSelect(el("va-wake-word"), (data.wake_words || []).map((w) => ({ value: w, label: w.replace(/_/g, " ") })));
      wireEngineAndDevice(el("va-engine"), el("va-compute-device"), data);
      wireLanguageModel(el("va-model"), data, { engine: el("va-engine"), device: el("va-compute-device") });
    },
    body() {
      return {
        audio_device: el("va-audio-device").value || null,
        engine: el("va-engine").value,
        compute_device: el("va-compute-device").value,
        llm_model: languageModelOf(el("va-model")),
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
    controls: ["expx-folder", "expx-sample", "expx-ocr-engine", "expx-ocr-device", "expx-llm-engine", "expx-llm-device", "expx-llm-model"],
    wireExtra() {
      this.review = new ExpenseReview({
        // The results left the view: so do the run's progress lines and its
        // "Read 5 receipts" status.
        onCleared: () => {
          this.resetProgress();
          if (!this.hasError()) this.setStatus("Idle");
        },
      });
    },
    resetProgress() {
      showPlaceholder(el("expx-transcript"), "Each receipt's vendor, date, amount, and category will appear here as it is structured.");
    },
    async rehydrate() {
      await StreamPanel.prototype.rehydrate.call(this);
      await this.review.refresh();
      clearInterval(this.reviewTimer);
      this.reviewTimer = setInterval(() => { if (this.review.data?.running) this.review.refresh(); }, 3000);
    },
    leave() {
      clearInterval(this.reviewTimer);
      StreamPanel.prototype.leave.call(this);
    },
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
      wireLanguageModel(el("expx-llm-model"), data, { engine: el("expx-llm-engine"), device: el("expx-llm-device") });
      wireSamplePicker("expx-sample", data.samples, { "expx-folder": "folder" });
    },
    body() {
      if (!this.review.canLeave()) throw new Error("Save your expense changes before starting a new batch.");
      const folder = el("expx-folder").value.trim();
      if (!folder) throw new Error("Enter a folder of receipt photos first.");
      rememberPath("expx-folder");
      return {
        folder,
        ocr_engine: el("expx-ocr-engine").value,
        ocr_compute_device: el("expx-ocr-device").value,
        llm_engine: el("expx-llm-engine").value,
        llm_compute_device: el("expx-llm-device").value,
        llm_model: languageModelOf(el("expx-llm-model")),
      };
    },
    onStarted(data) {
      this.review.selected = null;
      this.review.refresh(data.report_id);
      this.resetProgress();
    },
    onMessage(message) {
      const box = el("expx-transcript");
      if (message.type === "ocr_progress") {
        appendLine(box, "line-note", `Reading receipt ${message.index}/${message.total}: ${message.file}`);
      } else if (message.type === "structured") {
        this.review.refresh();
        const line = message.line;
        if (line.error) {
          appendLine(box, "line-answer", `${line.source_file}: ready for manual completion (${line.error})`);
        } else {
          const amount = line.amount !== null && line.amount !== undefined ? `${line.currency || "Unknown currency"} ${line.amount}` : "Unknown amount";
          const review = line.needs_review ? ` -- Needs review: ${(line.review_reasons || []).join("; ")}` : " -- Fields validated; verify against receipt";
          appendLine(box, "line-answer", `${line.source_file}: ${line.vendor || "?"} -- ${line.date || "?"} -- ${amount} -- ${line.category}${review}`);
        }
      } else if (message.type === "done") {
        this.setRunning(false);
        this.review.refresh();
        this.setStatus(`Read ${message.count} receipts. Review and validate each expense below.`, "live");
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
    controls: ["objdet-source", "objdet-source-device", "objdet-sample", "objdet-path", "objdet-engine", "objdet-compute-device"],
    populate(data) {
      const source = el("objdet-source");
      wireVideoSource(source, el("objdet-source-device"), data);
      wireEngineAndDevice(el("objdet-engine"), el("objdet-compute-device"), data, { preferRealtimeVision: true });
      const showSource = () => {
        const file = source.value === "file";
        el("objdet-path-field").hidden = !file;
        el("objdet-sample-field").hidden = !file;
        el("objdet-device-field").hidden = file;
      };
      source.addEventListener("change", showSource);
      showSource();

      const samples = data.samples || [];
      const picker = el("objdet-sample");
      for (const sample of samples) {
        picker.appendChild(option(sample.name, sample.name + (sample.ready === false ? " (fetched the first time it is started)" : "")));
      }
      picker.title = samples.map((sample) => `${sample.name}: ${sample.description}`).join("\n");
      picker.addEventListener("change", () => {
        const sample = samples.find((entry) => entry.name === picker.value);
        if (sample) el("objdet-path").value = sample.path;
      });
      // The panel opens ready to run, on a video kept on this machine: a
      // screen full of windows has nothing in it for a detector to find.
      const usual = samples.find((sample) => sample.default);
      if (usual && !el("objdet-path").value) {
        el("objdet-path").value = usual.path;
        picker.value = usual.name;
      }
    },
    body() {
      const source = el("objdet-source").value;
      const device = el("objdet-source-device").value;
      const path = el("objdet-path").value.trim();
      if (source === "file" && !path) throw new Error("Choose a sample video, or give the path of a video file.");
      return {
        source,
        path: source === "file" ? path : "",
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

  // A model watches a video and says what is happening; another gives the
  // line a mood. The mood is the one control left alive while it runs.
  "video-commentary": new StreamPanel({
    id: "video-commentary",
    prefix: "vidcom",
    transport: "mjpeg",
    video: "vidcom-video",
    statusKey: "video-commentary",
    controls: ["vidcom-source", "vidcom-source-device", "vidcom-sample", "vidcom-path", "vidcom-vision-device", "vidcom-mood-device", "vidcom-mood-model"],
    lastComment: 0,
    moodNames: {},
    sound: null, // the line being said
    // Whether the voices can be used is asked again each time the panel
    // opens: a voice may have been enrolled in the Voice Clone Studio since.
    async refreshVoices() {
      let voices = [];
      try {
        voices = (await fetchJSON("/api/video-commentary/devices")).voices || [];
      } catch {
        return;
      }
      const select = el("vidcom-voice");
      const chosen = select.value;
      fillSelect(select, [
        { value: "", label: "No: written only" },
        ...voices.map((voice) => ({
          value: voice.key,
          label: voice.ready ? `${voice.name} (${deviceLabel(voice.device)})` : `${voice.name}: enrol one in Voice Clone Studio first`,
        })),
      ]);
      for (const voice of voices) select.querySelector(`option[value="${voice.key}"]`).disabled = !voice.ready;
      select.value = voices.some((voice) => voice.key === chosen && voice.ready) ? chosen : "";
      this.voiceNote();
    },
    voiceNote(notice) {
      const note = el("vidcom-voice-note");
      const voice = el("vidcom-voice").value;
      const text =
        notice ||
        (voice === "cloned"
          ? "A cloned voice takes twenty to thirty seconds to make each line while the video plays: about two lines a minute, each well behind its picture, and Stop waits for the line in hand."
          : voice
            ? "Each line is spoken about three seconds after the picture it is about, and the next look waits for it to have been said: a line every eight to ten seconds."
            : "");
      note.textContent = text;
      note.hidden = !text;
    },
    hush() {
      if (!this.sound) return;
      this.sound.pause();
      this.sound = null;
    },
    say(number) {
      this.hush();
      const sound = new Audio(`/api/video-commentary/speech/${number}`);
      this.sound = sound;
      // A browser that will not play without a click says so by refusing: the line stays written.
      sound.play().catch(() => {});
    },
    async open() {
      await StreamPanel.prototype.open.call(this);
      this.refreshVoices();
    },
    leave() {
      this.hush();
      StreamPanel.prototype.leave.call(this);
    },
    populate(data) {
      const devices = data.openvino_devices || [];
      const auto = data.auto_devices || {};
      const fill = (id, key, allowed) =>
        fillSelect(el(id), [
          { value: "AUTO", label: auto[key] ? `Auto (${autoDeviceName(auto[key])})` : deviceLabel("AUTO") },
          ...devices.filter(allowed).map((d) => ({ value: d, label: deviceLabel(d) })),
        ]);
      // The vision model does not compile for the NPU on this hardware.
      fill("vidcom-vision-device", "vision", (d) => !d.toUpperCase().startsWith("NPU"));
      fill("vidcom-mood-device", "mood", () => true);
      wireLanguageModel(el("vidcom-mood-model"), data, { device: el("vidcom-mood-device"), autoDevice: () => auto.mood });
      fillSelect(el("vidcom-mood"), (data.moods || []).map((mood) => ({ value: mood.key, label: mood.name })));
      this.moodNames = Object.fromEntries((data.moods || []).map((mood) => [mood.key, mood.name]));
      if (data.default_mood) el("vidcom-mood").value = data.default_mood;
      el("vidcom-mood").addEventListener("change", () => {
        // While it runs, the next comment is in the new voice; before, it is just the choice Start will send.
        if (this.running) postJSON("/api/video-commentary/mood", { mood: el("vidcom-mood").value }).catch(() => {});
      });
      el("vidcom-voice").addEventListener("change", async () => {
        this.voiceNote();
        if (!el("vidcom-voice").value) this.hush();
        if (!this.running) return;
        try {
          await postJSON("/api/video-commentary/voice", { voice: el("vidcom-voice").value });
        } catch (err) {
          this.voiceNote(err.message);
        }
      });

      const source = el("vidcom-source");
      const showSource = () => {
        const file = source.value === "file";
        el("vidcom-path-field").hidden = !file;
        el("vidcom-sample-field").hidden = !file;
        el("vidcom-device-field").hidden = file;
        if (file) return;
        const items =
          source.value === "webcam"
            ? (data.cameras || []).map((index) => ({ value: String(index), label: `Camera ${index}` }))
            : (data.screens || []).map((s) => ({ value: String(s.index), label: `Screen ${s.index} (${s.width}x${s.height})` }));
        fillSelect(el("vidcom-source-device"), items.length ? items : [{ value: "", label: source.value === "webcam" ? "No camera found" : "No screen found" }]);
      };
      source.addEventListener("change", showSource);
      showSource();

      const samples = data.samples || [];
      const picker = el("vidcom-sample");
      for (const sample of samples) {
        picker.appendChild(option(sample.name, sample.name + (sample.ready === false ? " (fetched the first time it is started)" : "")));
      }
      picker.title = samples.map((sample) => `${sample.name}: ${sample.description}`).join("\n");
      picker.addEventListener("change", () => {
        const sample = samples.find((entry) => entry.name === picker.value);
        if (sample) el("vidcom-path").value = sample.path;
      });
      // The panel opens ready to run, on a video kept on this machine.
      const usual = samples.find((sample) => sample.default);
      if (usual && !el("vidcom-path").value) {
        el("vidcom-path").value = usual.path;
        picker.value = usual.name;
      }
      if (!devices.length) {
        el("vidcom-start").disabled = true;
        this.setStatus("The commentator needs the OpenVINO engine, which is not installed here.", "error");
      }
    },
    body() {
      const source = el("vidcom-source").value;
      const device = el("vidcom-source-device").value;
      const path = el("vidcom-path").value.trim();
      if (source === "file" && !path) throw new Error("Choose a sample video, or give the path of a video file.");
      return {
        source,
        path,
        camera_index: source === "webcam" ? Number(device || 0) : 0,
        screen_index: source === "screen" ? Number(device || 1) : 1,
        vision_device: el("vidcom-vision-device").value,
        mood_device: el("vidcom-mood-device").value,
        mood_model: languageModelOf(el("vidcom-mood-model")),
        mood: el("vidcom-mood").value,
        voice: el("vidcom-voice").value,
      };
    },
    onRunning(isRunning) {
      const log = el("vidcom-log");
      const caption = el("vidcom-caption");
      if (!isRunning) {
        caption.hidden = true;
        this.hush();
        return; // the list stays: what was said is still worth reading once it has stopped
      }
      // A run found already going (a reload, another page): its last lines are
      // listed, and not said again.
      let heardFrom = this.lastComment === 0 && STATUS.snapshot[this.statusKey] ? Infinity : 0;
      this.lastComment = 0;
      showPlaceholder(log, "The comments will be listed here as they come.");
      this.every(700, async () => {
        try {
          const data = await fetchJSON(`/api/video-commentary/comments?after=${this.lastComment}`);
          if (data.error) {
            this.setStatus(`Error: ${data.error}`, "error");
            this.setRunning(false);
            return;
          }
          // A mood changed from another page, or kept from before a reload.
          if (data.mood && document.activeElement !== el("vidcom-mood")) el("vidcom-mood").value = data.mood;
          if (data.notice) this.voiceNote(data.notice);
          const comments = data.comments || [];
          if (heardFrom === Infinity) heardFrom = comments.length ? comments[comments.length - 1].number : 0;
          // Of several lines come at once, only the last is said.
          const toSay = comments.filter((comment) => comment.speech && comment.number > heardFrom).pop();
          if (toSay && el("vidcom-voice").value) this.say(toSay.number);
          for (const comment of comments) {
            if (!this.lastComment) log.replaceChildren();
            this.lastComment = comment.number;
            const plain = comment.said === comment.seen;
            el("vidcom-said").textContent = comment.said;
            el("vidcom-seen").textContent = plain ? "" : `Seen: ${comment.seen}`;
            caption.hidden = false;
            const took =
              `${comment.seeing_seconds.toFixed(1)} s to see` +
              (plain ? "" : ` + ${comment.saying_seconds.toFixed(1)} s to say`) +
              (comment.speech ? ` + ${comment.voicing_seconds.toFixed(1)} s for the voice` : "");
            const entry = document.createElement("div");
            entry.className = "vidcom-entry";
            entry.innerHTML =
              `<p class="vidcom-entry-said">${escapeHtml(comment.said)}</p>` +
              `<p class="vidcom-entry-meta">${escapeHtml(this.moodNames[comment.mood] || comment.mood)} · ${escapeHtml(took)}` +
              (plain ? "" : `<br />Seen: ${escapeHtml(comment.seen)}`) +
              `</p>`;
            log.prepend(entry);
            while (log.children.length > 30) log.lastChild.remove();
          }
        } catch {
          // Best-effort -- a missed look shouldn't interrupt the video.
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
    renderHealth(health) {
      const cards = feedCards();
      this.feeds.forEach((feed, i) => {
        const entry = health[feed.feed_id];
        const card = cards[i];
        if (!card || !entry) return;
        let status = card.querySelector('.feed-health');
        if (!status) {
          status = document.createElement('p');
          status.className = 'feed-health';
          status.setAttribute('role', 'status');
          card.appendChild(status);
        }
        status.textContent = entry.message + (entry.frame_age_seconds == null ? '' : ` Last frame: ${Math.floor(entry.frame_age_seconds)}s ago.`);
      });
    },
    onStatus() {
      // Feed 1 is not the whole workload: another feed may still be healthy.
      if (!this.isOpen || !this.running) return;
      const stages = Object.values(STATUS.snapshot).filter(s => s.demo_id.startsWith('smart-city-monitor:'));
      const running = stages.filter(s => s.phase === 'running').length;
      const failed = stages.filter(s => s.phase === 'error').length;
      const stopping = stages.some(s => s.phase === 'stopping');
      this.setStatus(stopping ? 'Stopping...' : `${running} feed(s) running, ${failed} failed; see each feed below.`, stopping ? 'stopping' : running ? 'running' : 'loading');
    },
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
      fillGroupedSelect(
        picker,
        samples,
        (entry) => entry.name,
        // A video still to be fetched says so: starting it is what fetches it.
        (entry) => `${entry.name} -- ${entry.description}` + (entry.ready === false ? " Fetched the first time it is started." : ""),
      );
      picker.disabled = !samples.length;
      const addSample = (sample) => {
        const lines = sample.feeds.split("\n").map((l) => l.trim()).filter(Boolean);
        lines.forEach((line, index) => {
          const pipe = line.lastIndexOf("|");
          addFeedCard({
            type: sample.kind === "file" ? "file" : "url",
            path: pipe === -1 ? line : line.slice(0, pipe),
            device: pipe === -1 ? null : line.slice(pipe + 1).trim(),
            // What this feed is watched for: traffic, a production line, a herd.
            counting: (sample.counting || [])[index] || "street",
          });
        });
      };
      picker.onchange = () => {
        const sample = samples.find((entry) => entry.name === picker.value);
        picker.value = "";
        if (sample) addSample(sample);
      };

      // The panel opens ready to run: the street videos kept on this machine,
      // one per chip. A demo that needs the network to start does not start
      // in a conference hall. Failing that, one empty card to fill in.
      if (!feedCards().length) {
        const usual = samples.find((entry) => entry.default);
        if (usual) addSample(usual);
        else addFeedCard({ type: "url" });
      }
      renumberFeeds();
    },
    async rehydrate() {
      // The feed list lives on the server (a run may predate this visit):
      // pick it up from the counts route, then the usual running check. The
      // route reports each feed's engine, chip and source, so the cards can
      // be rebuilt exactly as they were rather than left blank under a run.
      let live = false;
      let lastError = null;
      try {
        const data = await fetchJSON("/api/smart-city-monitor/counts");
        live = Boolean(data.running);
        lastError = data.error;
        this.feeds = data.feeds || [];
        if (this.feeds.length) {
          el("smartcity-feed-list").innerHTML = "";
          for (const feed of this.feeds) {
            addFeedCard({
              type: feed.source_type || "url",
              path: feed.path,
              engine: feed.engine,
              device: feed.compute_device,
              counting: feed.counting,
            });
          }
          this.renderHealth(data.health || {});
        }
      } catch {
        this.feeds = [];
      }
      this.setRunning(live);
      if (lastError) this.setStatus(`Error: ${lastError}`, 'error');
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
          this.renderHealth(data.health || {});
          if (data.error) {
            this.setStatus(`Error: ${data.error}`, "error");
            this.setRunning(false);
            return;
          }
          if (data.running === false) {
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
    // Which files were read, by name: "it says it looked at three files and
    // I gave it two" has to be something a person can check.
    indexedText(result) {
      const files = result.files || [];
      const names = files.slice(0, 6).join(", ") + (files.length > 6 ? ` and ${files.length - 6} more` : "");
      const count = `${files.length} file${files.length === 1 ? "" : "s"}, ${result.chunks} passage${result.chunks === 1 ? "" : "s"}`;
      return `Indexed ${count} from ${result.folder}` + (names ? `: ${names}` : "");
    },
    populate(data) {
      attachRecents("docqa-folder");
      wireEngineAndDevice(el("docqa-engine"), el("docqa-compute-device"), data);
      wireLanguageModel(el("docqa-model"), data, { engine: el("docqa-engine"), device: el("docqa-compute-device") });
      this.modelNames = Object.fromEntries(((data.language_models || {}).models || []).map((model) => [model.key, model.name]));
      wireSamplePicker("docqa-sample", data.samples, { "docqa-folder": "folder", "docqa-question": "question" });
    },
    async rehydrate() {
      // An index built on a previous visit (or before a reload) is still
      // loaded server-side -- reflect it instead of asking to re-index.
      try {
        const status = await fetchJSON("/api/doc-qa/status");
        this.setIndexed(status.indexed);
        if (status.indexed) {
          this.setStatus(this.indexedText(status), "live");
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
      for (const id of ["docqa-folder", "docqa-engine", "docqa-compute-device", "docqa-model", "docqa-reindex", "docqa-sample"]) {
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
              model: languageModelOf(el("docqa-model")),
              reindex: el("docqa-reindex").checked,
            }),
          done: (r) => this.indexedText(r),
        });
        if (result) rememberPath("docqa-folder");
        this.setIndexed(Boolean(result));
      });
      const ask = async () => {
        const question = el("docqa-question").value.trim();
        if (!question) return;
        const box = el("docqa-transcript");
        // The same model with nothing to read: ask twice, once each way, and
        // what the documents change is on screen.
        const alone = el("docqa-alone").checked;
        appendLine(box, "line-question", `Q: ${question}` + (alone ? "  (without the documents)" : ""));
        el("docqa-question").value = alone ? question : ""; // kept, to be asked again with them
        this.setBusy(true);
        // The answer appears here as it is written, then gives way to the
        // finished one.
        const writing = document.createElement("div");
        box.appendChild(writing);
        const stopFollowing = followPartial("doc-qa", {
          target: writing,
          onUpdate: () => {
            box.scrollTop = box.scrollHeight;
          },
        });
        try {
          // The model can be changed between two questions: the index stays.
          const answer = await postJSON("/api/doc-qa/ask", { question, alone, model: languageModelOf(el("docqa-model")) });
          stopFollowing();
          writing.remove();
          appendLine(box, "line-answer", answer.text);
          if (answer.cancelled) appendLine(box, "line-note", "Stopped -- this answer is incomplete.");
          if (alone) {
            appendLine(box, "line-note", "The model alone: no document was in the conversation. Untick the box and ask again to see what they change.");
            el("docqa-alone").checked = false;
          }
          if (answer.sources && answer.sources.length) {
            // By file, with how many passages of each: what the answer was
            // written from, in a form that can be checked against the folder.
            const passages = new Map();
            for (const source of answer.sources) passages.set(source.source, (passages.get(source.source) || 0) + 1);
            const count = answer.sources.length;
            appendLine(
              box,
              "line-note",
              `Written ${(this.modelNames || {})[answer.model] ? `by ${this.modelNames[answer.model]} ` : ""}from ${count} passage${count === 1 ? "" : "s"} of ${passages.size} file${passages.size === 1 ? "" : "s"}: ` +
                [...passages].map(([file, times]) => (times > 1 ? `${file} (${times})` : file)).join(", "),
            );
          }
        } catch (err) {
          appendLine(box, "line-answer", `Error: ${err.message}`);
        } finally {
          stopFollowing();
          writing.remove();
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
          partial: { target: el("ocr-result") },
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
      container.innerHTML = stoppedNoteHtml(data) + html + generationStatsHtml(data.stats);
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
              "timbre rather than a person's full identity -- but it has delivery styles, and an OpenVINO " +
              "engine. Fastest on the CPU; a GPU compiles it again for every new sentence, and the NPU cannot run it.",
      },
    },
    modelInfo() {
      return this.MODEL_INFO[el("voice-model").value] || this.MODEL_INFO.chatterbox;
    },
    wireModelChoice() {
      const model = el("voice-model");
      const engine = el("voice-engine");
      const availableEngines = new Set([...engine.options].filter((option) => !option.disabled).map((option) => option.value));
      const apply = () => {
        const info = this.modelInfo();
        el("voice-model-help").textContent = info.help;
        // A model with one engine has nothing to choose: hide the pair
        // rather than offering a control whose only value is forced.
        const oneEngine = info.engines.length === 1;
        el("voice-engine-field").hidden = oneEngine;
        el("voice-device-field").hidden = oneEngine;
        for (const option of engine.options) {
          option.disabled = !availableEngines.has(option.value) || !info.engines.includes(option.value);
        }
        if (!info.engines.includes(engine.value) || engine.selectedOptions[0]?.disabled) {
          engine.value = [...engine.options].find((option) => !option.disabled)?.value || "";
        }
        // Programmatic engine selection must also rebuild the device list.
        engine.dispatchEvent(new Event("change"));
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
          partial: { target: el("cra-result") },
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
      el("cra-result").innerHTML = stoppedNoteHtml(data) + generationStatsHtml(data.stats) + html;
    },
  }),

  "html-creator": new Panel({
    id: "html-creator",
    prefix: "htmlc",
    currentHtml: null,
    populate(data) {
      attachRecents("htmlc-folder");
      attachRecents("htmlc-pictures");
      const mode = el("htmlc-mode");
      const sync = () => {
        const isLandingPage = mode.value === "landing_page";
        el("htmlc-prompt-field").hidden = !isLandingPage;
        el("htmlc-pictures-field").hidden = !isLandingPage;
        el("htmlc-folder-field").hidden = isLandingPage;
      };
      mode.addEventListener("change", sync);
      sync();
      wireEngineAndDevice(el("htmlc-engine"), el("htmlc-compute-device"), data, { preferLargeModel: true });
      wireSamplePicker(
        "htmlc-sample",
        data.samples,
        { "htmlc-mode": "mode", "htmlc-prompt": "prompt", "htmlc-folder": "folder" },
        // A sample without pictures must not inherit the previous one's folder.
        (sample) => {
          if (sample) el("htmlc-pictures").value = sample.pictures || "";
        },
      );
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
        const pictures = mode === "landing_page" ? el("htmlc-pictures").value.trim() : "";
        if (pictures) rememberPath("htmlc-pictures");
        el("htmlc-download").hidden = true;
        el("htmlc-fullscreen").hidden = true;
        this.run({
          button: el("htmlc-generate"),
          key: "html-creator",
          partial: { target: el("htmlc-result") },
          busy: "Generating…",
          work: async () => {
            const data = await postJSON("/api/html-creator/generate", {
              mode,
              prompt: el("htmlc-prompt").value,
              folder: el("htmlc-folder").value,
              pictures: pictures || null,
              repeatable: el("htmlc-repeatable").checked,
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
      el("htmlc-fullscreen").addEventListener("click", () => this.expand(true));
      el("htmlc-exit-fullscreen").addEventListener("click", () => this.expand(false));
      document.addEventListener("fullscreenchange", () => {
        if (!document.fullscreenElement) this.expand(false); // Escape left the browser's full screen
      });
    },
    // The page at the size it was designed for. Still the sandboxed frame: it
    // fills the window, and the whole screen where the browser grants that.
    expand(on) {
      const frame = el("htmlc-result").querySelector(".htmlc-preview-frame");
      if (!frame) on = false;
      if (frame) frame.classList.toggle("expanded", on);
      document.body.classList.toggle("htmlc-expanded", on);
      el("htmlc-exit-fullscreen").hidden = !on;
      if (on) document.documentElement.requestFullscreen?.().catch(() => {});
      else if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    },
    leave() {
      this.expand(false);
      Panel.prototype.leave.call(this);
    },
    renderResult(data) {
      this.currentHtml = data.html;
      const container = el("htmlc-result");
      container.innerHTML = stoppedNoteHtml(data) + generationStatsHtml(data.stats);
      if (data.html_truncated) {
        container.insertAdjacentHTML("beforeend", '<p class="section-label">Output doesn\'t end with </html> -- it may have been cut off.</p>');
      }
      if (data.source_truncated) {
        container.insertAdjacentHTML("beforeend", `<p class="section-label">Source was ${data.source_char_count} characters -- truncated before generation, some content may not be reflected.</p>`);
      }
      if (data.repeatable) {
        container.insertAdjacentHTML(
          "beforeend",
          '<p class="section-label">Written on a freshly loaded model -- the same prompt gives this page again.</p>',
        );
      }
      if (data.pictures_offered) {
        const notes = (data.picture_notes || []).map((note) => ` ${note}`).join("");
        container.insertAdjacentHTML(
          "beforeend",
          `<p class="section-label">${data.pictures_used.length} of ${data.pictures_offered} pictures placed and embedded in the page.${escapeHtml(notes)}</p>`,
        );
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
      // As the model wrote it: with pictures embedded, the page is megabytes
      // of base64 nobody can read.
      pre.textContent = data.html_source || data.html;
      details.appendChild(pre);
      container.appendChild(details);
      el("htmlc-download").hidden = false;
      el("htmlc-fullscreen").hidden = false;
    },
  }),

  "page-agent": new Panel({
    id: "page-agent",
    prefix: "agent",
    currentHtml: null,
    poll: null,
    populate(data) {
      const devices = data.openvino_devices || [];
      const auto = data.auto_assignment || {};
      // Each chip is the conductor's to choose unless one is named here, and
      // "Auto" says which it would be on this machine.
      const fill = (id, key, allowed) => {
        const label = auto[key] ? `Auto (${autoDeviceName(auto[key])})` : deviceLabel("AUTO");
        fillSelect(el(id), [
          { value: "AUTO", label },
          ...devices.filter(allowed).map((d) => ({ value: d, label: deviceLabel(d) })),
        ]);
      };
      const notNpu = (d) => !d.toUpperCase().startsWith("NPU");
      fill("agent-planner-device", "planner", () => true);
      fill("agent-image-device", "images", notNpu);
      fill("agent-page-device", "page", notNpu);
      wireSamplePicker("agent-sample", data.samples, { "agent-request": "prompt" });
      if (!devices.length) {
        el("agent-build").disabled = true;
        this.setStatus("The page agent needs the OpenVINO engine, which is not installed here.", "error");
      }
    },
    async rehydrate() {
      // A build started before this visit (or before a reload) is still
      // going, or left its plan and pictures behind: show where it is.
      try {
        const state = await fetchJSON("/api/page-agent/progress");
        if (state.run) this.renderProgress(state);
        if (state.running) this.follow();
      } catch {
        // Nothing to show is not an error worth a message.
      }
    },
    leave() {
      this.expand(false);
      clearInterval(this.poll);
      this.poll = null;
      Panel.prototype.leave.call(this);
    },
    follow() {
      clearInterval(this.poll);
      this.poll = setInterval(async () => {
        try {
          const state = await fetchJSON("/api/page-agent/progress");
          this.renderProgress(state);
          if (!state.running) {
            clearInterval(this.poll);
            this.poll = null;
          }
        } catch {
          // A missed ask: the next one has it all.
        }
      }, 700);
    },
    renderProgress(state) {
      const steps = el("agent-steps");
      steps.hidden = !state.steps.length;
      steps.innerHTML = state.steps
        .map((step, index) => {
          const seconds = step.seconds !== null && step.seconds !== undefined ? ` · ${step.seconds.toFixed(0)} s` : "";
          const what =
            { pending: "waiting", loading: "loading the model", running: step.detail || "working", done: step.detail || "done", failed: "failed" }[
              step.state
            ] || step.state;
          // How fast this step works, in its own unit: tokens per second for
          // the two language models, images per minute for the image model.
          const rate = step.rate && step.state !== "loading" ? `<div class="step-rate">${escapeHtml(formatMetric(step.rate))}</div>` : "";
          return (
            `<li class="agent-step ${escapeHtml(step.state)}"><div class="step-head">` +
            `<span class="step-name">${index + 1}. ${escapeHtml(step.label)}</span>` +
            `<span class="step-chip" title="${escapeHtml(deviceLabel(step.device))}">${escapeHtml(shortGpuName(deviceLabel(step.device)))}</span></div>` +
            `<div class="step-state">${escapeHtml(what)}${seconds}</div>${rate}</li>`
          );
        })
        .join("");
      const box = el("agent-plan");
      box.hidden = !state.plan;
      if (!state.plan) return;
      const plan = state.plan;
      const pictures = plan.pictures
        .map((picture) => {
          const inside = picture.ready
            ? `<img src="/api/page-agent/picture/${encodeURIComponent(picture.name)}?run=${state.run}" alt="${escapeHtml(picture.prompt)}" />`
            : "to be drawn";
          // An extra one: not in the plan, drawn because the page showed a
          // planned picture twice, or asked for one nobody had drawn, and said
          // in its alt text what belonged there.
          const took = (picture.seconds ? ` · ${picture.seconds.toFixed(1)} s` : "") + (picture.extra ? " · asked for by the page" : "");
          // Where the page is asked to put it: the plan gives each picture a place.
          const place = {
            hero: "behind the headline",
            offer: "on a card",
            story: "beside the story",
            closing: "behind the last call to action",
            feature: "beside text",
          }[picture.role];
          return (
            `<figure class="agent-picture"><div class="frame">${inside}</div>` +
            `<figcaption><strong>${escapeHtml(picture.name)}</strong> ${picture.width}×${picture.height}${took}` +
            (place ? `<br /><em>${place}</em>` : "") +
            `<br />${escapeHtml(picture.prompt)}</figcaption></figure>`
          );
        })
        .join("");
      const offers = plan.offers || [];
      box.innerHTML =
        (plan.title ? `<p class="plan-title">${escapeHtml(plan.title)}</p>` : "") +
        (plan.headline ? `<p class="plan-headline">${escapeHtml(plan.headline)}</p>` : "") +
        (plan.style ? `<p class="plan-line">${escapeHtml(plan.style)}</p>` : "") +
        (offers.length ? `<p class="plan-line">On offer: ${offers.map(escapeHtml).join(" · ")}</p>` : "") +
        (plan.sections.length ? `<p class="plan-line">${plan.sections.map(escapeHtml).join(" · ")}</p>` : "") +
        plan.notes.map((note) => `<p class="plan-note">${escapeHtml(note)}</p>`).join("") +
        `<div class="agent-pictures">${pictures}</div>`;
    },
    wire() {
      el("agent-build").addEventListener("click", () => {
        if (!el("agent-request").value.trim()) {
          this.setStatus("Say what the page is for first.", "error");
          return;
        }
        el("agent-download").hidden = true;
        el("agent-fullscreen").hidden = true;
        el("agent-plan").hidden = true;
        this.follow();
        this.run({
          button: el("agent-build"),
          key: "page-agent",
          partial: { target: el("agent-result"), stage: "page" },
          busy: "Starting…",
          work: async () => {
            try {
              const data = await postJSON("/api/page-agent/build", {
                request: el("agent-request").value,
                planner_device: el("agent-planner-device").value,
                image_device: el("agent-image-device").value,
                page_device: el("agent-page-device").value,
              });
              this.renderResult(data);
              return data;
            } finally {
              // One last look: the steps' final states and times.
              fetchJSON("/api/page-agent/progress").then((state) => this.renderProgress(state)).catch(() => {});
            }
          },
          done: (data) =>
            `Built in ${data.seconds.total.toFixed(0)} s` +
            (data.attempts > 1 ? " -- the page was written twice: the first one failed a check" : ""),
        });
      });
      el("agent-download").addEventListener("click", () => {
        if (!this.currentHtml) return;
        const url = URL.createObjectURL(new Blob([this.currentHtml], { type: "text/html" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = "page.html";
        link.click();
        URL.revokeObjectURL(url);
      });
      el("agent-fullscreen").addEventListener("click", () => this.expand(true));
      el("agent-exit-fullscreen").addEventListener("click", () => this.expand(false));
      document.addEventListener("fullscreenchange", () => {
        if (!document.fullscreenElement) this.expand(false);
      });
    },
    // As HTML Creator's: the page at the size it was designed for, still in
    // its sandboxed frame.
    expand(on) {
      const frame = el("agent-result").querySelector(".htmlc-preview-frame");
      if (!frame) on = false;
      if (frame) frame.classList.toggle("expanded", on);
      document.body.classList.toggle("htmlc-expanded", on);
      el("agent-exit-fullscreen").hidden = !on;
      if (on) document.documentElement.requestFullscreen?.().catch(() => {});
      else if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    },
    // What the build took and how fast each model worked, one row per
    // step, each in its own unit: an image model never produces a token.
    speedsHtml(data) {
      const chip = (device) => escapeHtml(shortGpuName(deviceLabel(device)));
      const loaded = (step) => {
        const seconds = (data.loads || {})[step];
        return seconds ? `<span class="perf-note">+ ${seconds.toFixed(0)} s loading the model</span>` : "";
      };
      const tokens = (stats) =>
        `<td><strong>${stats.tokens_per_second.toFixed(1)} tok/s</strong>` +
        (stats.first_token_seconds !== null && stats.first_token_seconds !== undefined
          ? `<span class="perf-note">first token after ${stats.first_token_seconds.toFixed(1)} s</span>`
          : "") +
        `</td>`;
      const rows = [];
      const plan = data.planner_stats;
      if (plan) {
        rows.push(
          `<tr><th scope="row">Plan</th><td>${chip(plan.device)}</td><td>${plan.tokens.toLocaleString()} tokens</td>` +
            `<td>${plan.seconds.toFixed(1)} s${loaded("plan")}</td>${tokens(plan)}</tr>`,
        );
      }
      const pictures = data.picture_stats;
      if (pictures) {
        const steps =
          pictures.steps_per_second !== null && pictures.steps_per_second !== undefined
            ? `<span class="perf-note">${pictures.steps_per_second.toFixed(2)} denoising steps/s</span>`
            : "";
        rows.push(
          `<tr><th scope="row">Pictures</th><td>${chip(pictures.device)}</td>` +
            `<td>${pictures.pictures} images<span class="perf-note">${pictures.megapixels.toFixed(1)} megapixels, ${pictures.steps} steps</span></td>` +
            `<td>${pictures.seconds.toFixed(1)} s${loaded("images")}</td>` +
            `<td><strong>${pictures.images_per_minute.toFixed(1)} images/min</strong>${steps}</td></tr>`,
        );
      }
      const page = data.stats;
      if (page) {
        const twice = data.attempts > 1 ? '<span class="perf-note">the second writing</span>' : "";
        rows.push(
          `<tr><th scope="row">Page</th><td>${chip(page.device)}</td><td>${page.tokens.toLocaleString()} tokens${twice}</td>` +
            `<td>${page.seconds.toFixed(1)} s${loaded("page")}</td>${tokens(page)}</tr>`,
        );
      }
      const how = data.assignment.together
        ? "the pictures were drawn while the page was written"
        : "the pictures were drawn first, then the page written";
      const energy = page && page.energy;
      const above = energy && energy.above_idle_joules !== null && energy.above_idle_joules !== undefined;
      const total =
        `<p class="gen-stats"><strong>${data.seconds.total.toFixed(0)} s</strong> in all -- ${escapeHtml(how)}` +
        (energy
          ? ` · <span class="energy" title="${escapeHtml(energyTitle(energy))}">${escapeHtml(energyText(energy))}` +
            `${above ? " above idle" : " (processor package)"}</span>`
          : "") +
        `</p>`;
      if (!rows.length) return total;
      return (
        `<div class="agent-perf-wrap"><table class="agent-perf"><thead><tr><th>Step</th><th>Chip</th><th>Work</th>` +
        `<th>Time</th><th>Speed</th></tr></thead><tbody>${rows.join("")}</tbody></table></div>${total}`
      );
    },
    renderResult(data) {
      this.currentHtml = data.html;
      const container = el("agent-result");
      const checks = data.checks
        .map(
          (check) =>
            `<li class="${check.passed ? "passed" : "failed"}">${check.passed ? "✓" : "✗"} ${escapeHtml(check.name)}` +
            `${check.detail && !check.passed ? ` -- ${escapeHtml(check.detail)}` : ""}</li>`,
        )
        .join("");
      container.innerHTML = stoppedNoteHtml(data) + `<ul class="agent-checks">${checks}</ul>` + this.speedsHtml(data);
      const iframe = document.createElement("iframe");
      iframe.className = "htmlc-preview-frame";
      iframe.setAttribute("sandbox", "allow-scripts");
      iframe.title = "Built page preview";
      iframe.srcdoc = data.html;
      container.appendChild(iframe);
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "View raw HTML";
      details.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "text-block";
      pre.textContent = data.html_source || data.html; // as written: the embedded pictures are megabytes of base64
      details.appendChild(pre);
      container.appendChild(details);
      el("agent-download").hidden = false;
      el("agent-fullscreen").hidden = false;
    },
  }),
};

// --- Home grid ----------------------------------------------------------------------

function renderBadges(container, demo) {
  container.innerHTML = "";
  if (demo.large_model) {
    const model = demo.large_model;
    const badge = document.createElement("span");
    badge.className = "badge badge-model";
    badge.textContent = `${model.label} · ${model.gpu_memory_gb} GB`;
    badge.title =
      `The OpenVINO engine runs a ${model.label} here. It takes about ${model.gpu_memory_gb} GB of GPU memory once loaded, ` +
      "which an integrated GPU draws from shared system memory -- no discrete GPU needed. " +
      (model.discrete_gpu ? `${model.discrete_gpu} ` : "") +
      `Measured on ${model.measured_on}.`;
    container.appendChild(badge);
  }
  if (demo.experimental) {
    const badge = document.createElement("span");
    badge.className = "badge badge-experimental";
    badge.textContent = "Experimental";
    badge.title = "It works, and it is still finding its shape: results vary from one build to the next, and its controls may change.";
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

// "#/" is the start screen, "#/stage" the Auto Demo's stage, "#/demos" the
// grid and "#/brick/<id>" a demo's panel. A reload stays where it was: only
// the address the launcher opens lands on the start screen.
function route() {
  if (location.hash === "#/stage") {
    showHome(); // whatever panel was open lets go of what it holds
    setScreen("stage");
    document.title = "Auto Demo · Panther Lake AI Studio";
    stageOpen();
    return;
  }
  stageClose();
  if (["", "#", "#/"].includes(location.hash)) {
    showLanding();
    return;
  }
  setScreen("studio");
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
  renderRunning();
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
  renderRunning();
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

// The header in one line, shown once the real one has scrolled out of view.
// (It used to carry the running demos as well; the hardware panel has them
// now, and it never scrolls away.) Asking the element where it is beats
// caching a scroll offset: it stays right when the window resizes or a
// wrapped header row changes height.
let compactFrame = 0;

function updateCompactBar() {
  const bar = el("compact-bar");
  const topbar = document.querySelector(".topbar");
  if (!bar || !topbar) return;
  bar.classList.toggle("is-visible", topbar.getBoundingClientRect().bottom <= 0);
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

// Each card on the grid says whether its brick is loading or running.
function renderCardStates() {
  const groups = new Map();
  for (const [key, entry] of Object.entries(STATUS.snapshot)) {
    const base = key.split(":")[0];
    if (!groups.has(base)) groups.set(base, []);
    groups.get(base).push(entry);
  }
  const phases = new Map();
  for (const [base, entries] of groups) {
    const status = summarizeStatus(entries);
    if (status) phases.set(base, status.phase);
  }
  for (const card of document.querySelectorAll(".card[data-id]")) {
    const phase = phases.get(card.dataset.id);
    card.classList.toggle("is-running", phase === "running");
    card.classList.toggle("is-loading", phase === "loading" || phase === "stopping");
    card.querySelector(".card-state").textContent = PHASE_LABEL[phase] || "";
  }
}

// Everything on the page that says what is running: the cards, and the
// bricks listed under each chip in the hardware panel.
function renderRunning() {
  renderCardStates();
  renderChipBricks();
}

async function pollStatus() {
  try {
    STATUS.snapshot = await fetchJSON("/api/status");
  } catch {
    return; // best-effort -- a missed poll just skips this tick
  }
  renderRunning();
  if (currentPanel) currentPanel.onStatus();
}

// --- Hardware panel ---------------------------------------------------------------
// Power first, then every chip in a fixed order -- CPU, integrated GPU,
// discrete GPU when there is one, NPU. Each shows its load and, under it,
// what is running on it: the brick's name, its own number in its own unit
// (tokens/s, frames/s, times real time), and a button to stop it.

const PANEL = { sections: [], markup: new Map(), stopping: new Set() };

function gpuKind(gpu) {
  const name = gpu.full_name || "";
  return name.includes("dGPU") ? "dGPU" : name.includes("iGPU") ? "iGPU" : "GPU";
}

function chipSectionsFor(gpus) {
  const integrated = gpus.filter((gpu) => gpuKind(gpu) !== "dGPU");
  const discrete = gpus.filter((gpu) => gpuKind(gpu) === "dGPU");
  return [
    { key: "CPU", label: "CPU", name: "" },
    ...integrated.map((gpu) => ({ key: gpu.id, label: gpuKind(gpu), name: shortGpuName(gpu.full_name) })),
    ...discrete.map((gpu) => ({ key: gpu.id, label: "dGPU", name: shortGpuName(gpu.full_name) })),
    { key: "NPU", label: "NPU", name: "" },
  ];
}

// The section a device belongs under. Nothing is ever dropped: a device
// that is none of the chips (cuda, say) gets an "Other" section. A brick
// left on "Auto" never reaches here as "AUTO" -- the launcher resolves it to
// a real chip before starting, which is what used to make such bricks
// vanish from the gauges.
function chipKeyFor(device) {
  const id = String(device || "").toUpperCase();
  const keys = PANEL.sections.map((section) => section.key);
  if (keys.includes(id)) return id;
  if (id.startsWith("GPU")) {
    // "GPU" on a machine that numbers its GPUs, or "GPU.0" on one that doesn't.
    const gpuKeys = keys.filter((key) => key.startsWith("GPU"));
    if (gpuKeys.length) return gpuKeys[0];
  }
  return "OTHER";
}

function chipNode(key) {
  return document.querySelector(`#chip-sections .chip-section[data-chip="${key}"]`);
}

function buildChipPanel() {
  PANEL.sections = chipSectionsFor(GPU_DEVICES);
  PANEL.markup.clear();
  const container = el("chip-sections");
  container.replaceChildren();
  for (const section of [...PANEL.sections, { key: "OTHER", label: "Other", name: "" }]) {
    const node = document.createElement("section");
    node.className = "chip-section";
    node.dataset.chip = section.key;
    node.dataset.count = "0";
    node.hidden = section.key === "OTHER";
    node.innerHTML =
      `<div class="chip-head"><span class="chip-label" data-short="${escapeHtml(section.label)}">${escapeHtml(section.label)}</span>` +
      `<span class="chip-name">${escapeHtml(section.name)}</span><span class="chip-value">--</span></div>` +
      `<div class="telemetry-bar"><div class="telemetry-bar-fill"></div></div>` +
      `<ul class="chip-bricks"></ul><p class="chip-empty">Nothing running</p>`;
    container.append(node);
  }
}

function renderTelemetry(data) {
  STATUS.active = data.active || [];
  STATUS.metrics = data.metrics || [];
  STATUS.loaded = data.loaded || [];
  STATUS.telemetry = data;
  if (STAGE.on) stageSide();

  const loads = { CPU: { value: data.cpu_percent }, NPU: { value: data.npu_percent, name: data.npu_name } };
  for (const gpu of data.gpus || []) loads[chipKeyFor(gpu.id)] = { value: gpu.percent };
  for (const section of PANEL.sections) {
    const node = chipNode(section.key);
    if (!node) continue;
    const load = loads[section.key] || {};
    const known = load.value !== null && load.value !== undefined;
    node.querySelector(".chip-value").textContent = known ? `${Math.round(load.value)}%` : "N/A";
    node.querySelector(".telemetry-bar-fill").style.width = known ? `${Math.min(load.value, 100)}%` : "0%";
    node.classList.toggle("unavailable", !known);
    if (load.name) node.querySelector(".chip-name").textContent = load.name.replace(/Intel\(R\)\s*/g, "");
  }
  renderNpuLost(data.npu_lost);
  renderPower(data.power);
  renderDeviceSummary(data);
  renderChipBricks();
}

// Windows can reset the NPU under a running model. From then on the app keeps
// off it until it is restarted (pantherlake_ai_core.npu), and its section says
// so in place of "Nothing running".
function renderNpuLost(lost) {
  const node = chipNode("NPU");
  if (!node) return;
  node.classList.toggle("lost", Boolean(lost));
  let text = "Nothing running";
  if (lost) {
    const when = lost.at ? new Date(lost.at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
    text = `Reset by Windows${when ? ` at ${when}` : ""}: out of use until the app is restarted. Demos set to the NPU run on ${deviceLabel(lost.moved_to)}.`;
  }
  const empty = node.querySelector(".chip-empty");
  if (empty.textContent !== text) empty.textContent = text;
}

// One row per running stage (a brick with two stages on two chips appears
// under both), plus the bricks that are idle but still hold a model.
function panelRows() {
  const rows = STATUS.active.map((entry) => ({
    demoId: entry.demo_id,
    stage: entry.stage || "default",
    stageLabel: entry.stage_label,
    device: entry.device,
    kind: "active",
    canStop: entry.can_stop !== false,
  }));
  for (const held of STATUS.loaded || []) {
    const stage = held.stage || "default";
    const atWork = rows.filter((row) => row.kind === "active" && row.demoId === held.demo_id);
    // A brick at work has no row for its idle model -- unless it holds several
    // models, one per chip (the page agent): then a step that has finished
    // keeps its row, and its last figure, while the others work. It cannot be
    // unloaded from there until they have finished.
    if (atWork.some((row) => stage === "default" || row.stage === stage)) continue;
    rows.push({
      demoId: held.demo_id,
      stage,
      stageLabel: held.stage_label || null,
      device: held.device,
      kind: "loaded",
      canStop: atWork.length === 0,
    });
  }
  return rows;
}

function rowStatus(row) {
  if (row.kind === "loaded") return { phase: "loaded", message: "Idle, with its model still loaded on this chip." };
  const key = row.stage === "default" ? row.demoId : `${row.demoId}:${row.stage}`;
  const entry = STATUS.snapshot[key] || STATUS.snapshot[row.demoId];
  return entry ? { phase: entry.phase, message: entry.message || "" } : { phase: "running", message: "" };
}

function formatMetric(metric) {
  const value = Number(metric.value);
  if (metric.unit === "fps") return `${value >= 10 ? Math.round(value) : value.toFixed(1)} fps`;
  if (metric.unit === "x real time") return `${value.toFixed(1)}× real time`;
  return `${value.toFixed(1)} ${metric.unit}`;
}

// A brick's own number, in its own unit. "last" marks one from a finished
// answer rather than work happening now.
function rowMetricHtml(row, status) {
  const mine = (STATUS.metrics || []).filter((metric) => metric.demo_id === row.demoId);
  const parts = [];
  const own = mine.find((metric) => metric.stage === row.stage);
  if (own) {
    const prefix = own.sticky || row.kind === "loaded" ? "last " : "";
    const suffix = !prefix && own.detail ? ` (${escapeHtml(own.detail)})` : "";
    parts.push(`${prefix}<strong>${escapeHtml(formatMetric(own))}</strong>${suffix}`);
  }
  // A finished stage's number (meeting notes' summary, say) rides on the
  // brick's main row, since that stage no longer has a row of its own.
  if (row.stage === "default") {
    for (const metric of mine) {
      if (metric === own || !metric.sticky) continue;
      parts.push(`${escapeHtml(metric.detail || metric.stage)} <strong>${escapeHtml(formatMetric(metric))}</strong>`);
    }
  }
  if (parts.length) return parts.join(" · ");
  if (status.phase === "loading") return "loading the model…";
  if (status.phase === "stopping") return "stopping…";
  return row.kind === "loaded" ? "model loaded, idle" : "";
}

function rowHtml(row) {
  const demo = demoById(row.demoId);
  const name = demo ? demo.name : row.demoId;
  const status = rowStatus(row);
  const busy = PANEL.stopping.has(row.demoId) || status.phase === "stopping";
  const current = currentPanel && currentPanel.id === row.demoId ? " current" : "";
  const verb = row.kind === "loaded" ? "Unload" : "Stop";
  const stopTitle = !row.canStop
    ? "It is still loading its model, or isn't writing an answer; it can be unloaded once that has finished."
    : row.kind === "loaded"
      ? `Unload ${name}'s model and free this chip's memory`
      : `Stop ${name}`;
  return (
    `<li class="chip-brick kind-${row.kind} phase-${escapeHtml(status.phase)}${current}">` +
    `<button type="button" class="chip-brick-open" data-open="${escapeHtml(row.demoId)}" title="${escapeHtml(status.message)}">` +
    `<span class="chip-dot"></span><span class="chip-brick-name">${escapeHtml(name)}</span>` +
    (row.stageLabel ? `<span class="chip-brick-stage">${escapeHtml(row.stageLabel)}</span>` : "") +
    `</button>` +
    `<button type="button" class="chip-brick-stop" data-stop="${escapeHtml(row.demoId)}" data-stage="${escapeHtml(row.stage)}" aria-label="${escapeHtml(`${verb} ${name}`)}" ` +
    `title="${escapeHtml(stopTitle)}"${busy || !row.canStop ? " disabled" : ""}>✕</button>` +
    `<span class="chip-brick-metric">${rowMetricHtml(row, status)}</span></li>`
  );
}

function renderChipBricks() {
  if (!PANEL.sections.length) return;
  const byChip = new Map();
  for (const row of panelRows()) {
    const key = chipKeyFor(row.device);
    if (!byChip.has(key)) byChip.set(key, []);
    byChip.get(key).push(row);
  }
  for (const key of [...PANEL.sections.map((section) => section.key), "OTHER"]) {
    const node = chipNode(key);
    if (!node) continue;
    const rows = byChip.get(key) || [];
    const markup = rows.map(rowHtml).join("");
    // Only touch the list when it changed: this runs twice a second, and
    // rewriting it would drop hover and focus from under the pointer.
    if (PANEL.markup.get(key) !== markup) {
      PANEL.markup.set(key, markup);
      node.querySelector(".chip-bricks").innerHTML = markup;
    }
    const running = new Set(rows.filter((row) => row.kind === "active").map((row) => row.demoId));
    node.classList.toggle("active", running.size > 0);
    node.classList.toggle("shared", running.size > 1);
    node.dataset.count = String(rows.length);
    if (key === "OTHER") node.hidden = rows.length === 0;
  }
}

// Watts for the whole processor package, from its RAPL energy counters
// (BACKLOG R18). Hidden -- never zero -- on a machine without them. The bar is
// scaled to the chip's 80 W maximum turbo power; the note carries the number
// that matters, how far above idle the running demos push it.
const POWER_SCALE_W = 80;

function renderPower(power) {
  const section = el("chip-power");
  if (!section) return;
  section.hidden = !(power && power.available);
  if (section.hidden) return;
  const fmt = (w) => (w === null || w === undefined ? "?" : `${w.toFixed(1)} W`);
  const hasIdle = power.idle_w !== null && power.idle_w !== undefined;
  el("chip-power-value").textContent = fmt(power.package_w);
  el("chip-power-fill").style.width = `${Math.min((power.package_w / POWER_SCALE_W) * 100, 100)}%`;
  const notes = [];
  if (hasIdle) notes.push(`+${Math.max(power.package_w - power.idle_w, 0).toFixed(1)} W over idle`);
  const battery = power.battery;
  if (battery && !battery.plugged) notes.push(`on battery, ${battery.percent}%`);
  el("chip-power-note").textContent = notes.join(" · ") || "processor package";
  let title =
    `Processor package ${fmt(power.package_w)}: CPU cores ${fmt(power.cores_w)}, graphics ${fmt(power.graphics_w)}, ` +
    `rest of the chip ${fmt(power.rest_w)} (the NPU, memory controller and I/O -- the NPU has no rail of its own). ` +
    `Memory ${fmt(power.memory_w)}.`;
  title += hasIdle ? ` Idle baseline ${fmt(power.idle_w)}.` : " Idle baseline not learned yet.";
  if (battery) {
    title += battery.plugged
      ? ` Battery ${battery.percent}%, plugged in.`
      : ` Battery ${battery.percent}%` + (battery.seconds_left ? `, about ${Math.round(battery.seconds_left / 60)} min left.` : ".");
  }
  section.title = title;
  section.classList.toggle("active", STATUS.active.length > 0);
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

// The close button on a row: stops a brick that runs a loop, unloads the
// model of one that only answers requests. The launcher knows which is which.
async function stopBrick(demoId, stage = "default") {
  if (PANEL.stopping.has(demoId)) return;
  PANEL.stopping.add(demoId);
  el("chip-panel-message").textContent = "";
  renderChipBricks();
  try {
    // A stage's own row stops that stage's answer (meeting notes' summary,
    // without ending the transcription); the launcher falls back to the
    // whole brick when the stage isn't writing one.
    const query = stage && stage !== "default" ? `?stage=${encodeURIComponent(stage)}` : "";
    await postJSON(`/api/bricks/${demoId}/stop${query}`, {});
  } catch (err) {
    const demo = demoById(demoId);
    el("chip-panel-message").textContent = `${demo ? demo.name : demoId}: ${err.message}`;
  } finally {
    PANEL.stopping.delete(demoId);
  }
  await Promise.all([pollStatus(), pollTelemetry()]);
}

function setChipPanelCollapsed(collapsed, { remember = true } = {}) {
  document.body.classList.toggle("chip-panel-collapsed", collapsed);
  const toggle = el("chip-panel-toggle");
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.setAttribute("aria-label", collapsed ? "Expand the hardware panel" : "Collapse the hardware panel");
  if (!remember) return;
  try {
    localStorage.setItem("ptl.chipPanel", collapsed ? "collapsed" : "open");
  } catch {
    // Private mode: the choice just doesn't outlive the page.
  }
}

function wireChipPanel() {
  let stored = null;
  try {
    stored = localStorage.getItem("ptl.chipPanel");
  } catch {
    stored = null;
  }
  // A rail by default where the page has no room to give up 300px.
  setChipPanelCollapsed(stored ? stored === "collapsed" : window.innerWidth < 1100, { remember: false });
  el("chip-panel-toggle").addEventListener("click", (event) => {
    event.stopPropagation();
    setChipPanelCollapsed(!document.body.classList.contains("chip-panel-collapsed"));
  });
  el("chip-panel").addEventListener("click", (event) => {
    // As a rail, the whole thing is the way back in.
    if (document.body.classList.contains("chip-panel-collapsed")) {
      setChipPanelCollapsed(false);
      return;
    }
    const stop = event.target.closest("[data-stop]");
    if (stop) {
      if (!stop.disabled) stopBrick(stop.dataset.stop, stop.dataset.stage);
      return;
    }
    const open = event.target.closest("[data-open]");
    if (open) location.hash = `#/brick/${open.dataset.open}`;
  });
}

async function initTelemetry() {
  try {
    GPU_DEVICES = await fetchJSON("/api/system/gpu-devices");
  } catch {
    GPU_DEVICES = [];
  }
  buildChipPanel();
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

// --- Start screen, and the Auto Demo's stage ---------------------------------------
// The app opens on a start screen with two ways in: the demos to drive
// yourself, or the Auto Demo -- the studio presenting itself on a stand with
// nobody at the keyboard. The launcher directs that (autodemo.py) and this
// page follows on a stage of its own: a caption that tells the demo in hand
// as a story, a sentence or two at a time; what that demo puts out, and
// nothing else; and the chips at work down the right. Nothing to set, nothing
// to scroll.

const AUTODEMO = {
  state: null,
  // `left_out`: the scenes unticked on the start screen, by key. What is
  // remembered is what was taken out, so that a scene added to the playlist
  // later plays without anybody having to find it and tick it.
  options: { dgpu: "auto", big_screen: false, lang: "en", camera: "auto", fullscreen: true, left_out: [] },
  listed: [], // the playlist as the last look at the stand gave it
};
const AUTODEMO_POLL_MS = 1000;

function autodemoActive() {
  return !!AUTODEMO.state && ["checking", "playing", "paused"].includes(AUTODEMO.state.state);
}

// What the stage itself says, around the story the scenes bring with them.
const STAGE_TEXT = {
  en: {
    tag: "Auto Demo",
    checking: "Looking at what this machine has…",
    next: "The next demo is about to start…",
    paused: "The Auto Demo is paused. It takes up again by itself.",
    stoppedTitle: "The Auto Demo has stopped.",
    back: "Back to the start screen",
    pausedPill: (clock) => `Paused · takes up again by itself in ${clock}`,
    pausingPill: "Pausing as soon as this demo has finished",
    chips: {},
    idle: "Not needed for this demo",
    starting: "at work…",
    power: "Power",
    overIdle: (watts) => `+${watts} W over idle`,
    popupTitle: "This demo is running by itself",
    popupText: "It goes round the studio's demos and tells each one as it runs. What would you like to do?",
    popupPaused: "It is paused for now. What would you like to do?",
    keep: "Keep playing",
    resume: "Resume",
    pause: "Pause",
    stay: "Stay paused",
    stop: "Stop the Auto Demo",
    skip: "Skip to the next demo",
    stopping: "Stopping what is running…",
    request: "The request",
    plan: "1 · The plan",
    pictures: "2 · The pictures",
    page: "3 · The page",
    waitingPlan: "The plan will appear here.",
    waitingPage: "The page will be written here.",
    writing: "being written",
    built: "built on this machine",
    onOffer: "On offer",
    receiptWaiting: "Waiting its turn",
    receiptReading: "Being read…",
    vendor: "Vendor",
    date: "Date",
    amount: "Amount",
    category: "Category",
    toCheck: "To check",
    reasons: {},
    receiptsDone: (count, total, flagged) => `${count} receipts read · ${total}` + (flagged ? ` · ${flagged} to check` : ""),
    connecting: "Connecting to the camera…",
    counted: "Counted so far",
    nothingYet: "nothing yet",
    loadingModels: "The two models are loading…",
    saw: "What it saw",
    saidTitle: "What was said",
    firstLine: "The first line is on its way…",
    toSee: (seconds) => `${seconds} s to see`,
    toSay: (seconds) => `${seconds} s to say`,
    camera: "This laptop's camera",
    startingDetector: "Starting the camera…",
    inPicture: "In the picture now",
    nothingSeen: "nothing right now",
    saidOfIt: "What the vision model says",
    folder: "The folder",
    notShown: "On this laptop's disk. Not in the conversation.",
    nowRead: "Read on this laptop. The passages closest to a question go into the conversation with it.",
    fileStates: { out: "not shown to the model", read: "read", used: "used for the answer" },
    askedAlone: "1 · Asked of the model alone",
    askedWith: "2 · Asked again, with the files",
    asking: "Asking…",
    notYet: "Once the folder has been read.",
    noFiles: "no file in the conversation",
    fromFiles: "written from the files",
    filesRead: (count, passages) => `↓  ${count} files read: ${passages} passages`,
  },
  fr: {
    tag: "Démo automatique",
    checking: "Inventaire de ce que cette machine possède…",
    next: "La prochaine démo va commencer…",
    paused: "La démo automatique est en pause. Elle reprendra d'elle-même.",
    stoppedTitle: "La démo automatique s'est arrêtée.",
    back: "Retour à l'écran d'accueil",
    pausedPill: (clock) => `En pause · reprise automatique dans ${clock}`,
    pausingPill: "Pause dès que cette démo sera terminée",
    chips: { "Integrated GPU": "GPU intégré" },
    idle: "Pas utile pour cette démo",
    starting: "au travail…",
    power: "Consommation",
    overIdle: (watts) => `+${watts} W par rapport au repos`,
    popupTitle: "Cette démo tourne toute seule",
    popupText: "Elle enchaîne les démos du studio et raconte chacune pendant qu'elle tourne. Que souhaitez-vous faire ?",
    popupPaused: "Elle est en pause pour l'instant. Que souhaitez-vous faire ?",
    keep: "Continuer",
    resume: "Reprendre",
    pause: "Mettre en pause",
    stay: "Rester en pause",
    stop: "Arrêter la démo automatique",
    skip: "Passer à la démo suivante",
    stopping: "Arrêt de ce qui tourne…",
    request: "La demande",
    plan: "1 · Le plan",
    pictures: "2 · Les images",
    page: "3 · La page",
    waitingPlan: "Le plan apparaîtra ici.",
    waitingPage: "La page s'écrira ici.",
    writing: "en cours d'écriture",
    built: "construite sur cette machine",
    onOffer: "Au programme",
    receiptWaiting: "En attente",
    receiptReading: "Lecture en cours…",
    vendor: "Fournisseur",
    date: "Date",
    amount: "Montant",
    category: "Catégorie",
    toCheck: "À vérifier",
    reasons: {
      "Amount could not be matched to the receipt text": "montant introuvable sur le reçu",
      "Amount is missing or ambiguous": "montant absent ou ambigu",
      "Amount is not the printed total": "le montant n'est pas le total imprimé",
      "Date can be read two ways": "date lisible de deux façons",
      "Currency is missing, unsupported or ambiguous": "devise absente ou ambiguë",
      "Date is missing or invalid": "date absente ou invalide",
      "Vendor is missing": "fournisseur absent",
      "Category is invalid": "catégorie invalide",
      "Fractional amount for a zero-decimal currency": "montant décimal pour une devise sans décimales",
      "No text detected by OCR": "aucun texte lu sur le reçu",
    },
    receiptsDone: (count, total, flagged) => `${count} reçus lus · ${total}` + (flagged ? ` · ${flagged} à vérifier` : ""),
    connecting: "Connexion à la caméra…",
    counted: "Comptés jusqu'ici",
    nothingYet: "rien pour l'instant",
    loadingModels: "Chargement des deux modèles…",
    saw: "Ce qu'il a vu",
    saidTitle: "Ce qui a été dit",
    firstLine: "La première phrase arrive…",
    toSee: (seconds) => `${seconds} s pour voir`,
    toSay: (seconds) => `${seconds} s pour dire`,
    camera: "La caméra de ce portable",
    startingDetector: "Démarrage de la caméra…",
    inPicture: "À l'image en ce moment",
    nothingSeen: "rien pour l'instant",
    saidOfIt: "Ce qu'en dit le modèle de vision",
    folder: "Le dossier",
    notShown: "Sur le disque de ce portable. Pas dans la conversation.",
    nowRead: "Lu sur ce portable. Les passages les plus proches d'une question entrent avec elle dans la conversation.",
    fileStates: { out: "pas montré au modèle", read: "lu", used: "utilisé pour la réponse" },
    askedAlone: "1 · Posée au modèle seul",
    askedWith: "2 · Posée à nouveau, avec les fichiers",
    asking: "Question en cours…",
    notYet: "Quand le dossier aura été lu.",
    noFiles: "aucun fichier dans la conversation",
    fromFiles: "rédigée à partir des fichiers",
    filesRead: (count, passages) => `↓  ${count} fichiers lus : ${passages} passages`,
  },
};

function stageText() {
  const lang = (AUTODEMO.state && AUTODEMO.state.stand && AUTODEMO.state.stand.lang) || AUTODEMO.options.lang;
  return STAGE_TEXT[lang] || STAGE_TEXT.en;
}

// ------------------------------------------------------------------ which screen is up

function setScreen(name) {
  document.body.classList.toggle("screen-landing", name === "landing");
  document.body.classList.toggle("screen-stage", name === "stage");
  el("landing").hidden = name !== "landing";
  el("stage").hidden = name !== "stage";
}

function showLanding() {
  showHome(); // whatever panel was open lets go of what it holds
  setScreen("landing");
  document.title = "Panther Lake AI Studio";
  landingRender();
}

function landingRender() {
  const state = AUTODEMO.state;
  const note = el("landing-notice");
  // The loop gave up by itself: whoever comes back to the machine is told why.
  const stopped = !!state && state.state === "stopped" && !!state.notice;
  // Told to stop, and not done yet: a page being planned ends when its
  // model has answered. It cannot be started again until then.
  const stopping = !!state && state.state === "stopping";
  note.hidden = !(stopped || stopping);
  if (stopped) note.textContent = state.notice;
  if (stopping) note.textContent = "The Auto Demo is stopping: the demo it was showing is finishing. It can be started again in a moment.";
  el("landing-auto").disabled = stopping;
}

// ------------------------------------------------------------------ before it starts

function autodemoStandHtml(stand) {
  const yes = (text) => `<li class="found">${escapeHtml(text)}</li>`;
  const no = (text) => `<li class="missing">${escapeHtml(text)}</li>`;
  return (
    `<ul class="autodemo-stand">` +
    (stand.npu ? yes("NPU") : no("No NPU")) +
    (stand.igpu ? yes(`Integrated GPU (${stand.igpu})`) : no("No integrated GPU")) +
    (stand.dgpu ? yes(`Discrete GPU (${stand.dgpu})`) : no("No discrete GPU in use")) +
    (stand.cameras
      ? yes(stand.cameras === 1 ? "A camera" : `${stand.cameras} cameras`)
      : no(stand.cameras_found ? "Camera not used" : "No camera")) +
    (stand.internet ? yes("Internet") : no("No internet")) +
    `</ul>`
  );
}

// Each scene with a tick: what plays is chosen here. One that cannot play on
// this stand says why, and cannot be ticked.
function autodemoPlaylistHtml(playlist, leftOut) {
  const out = new Set(leftOut);
  const playing = (key) => {
    const entry = playlist.find((other) => other.key === key);
    return !!entry && entry.playable && !out.has(key);
  };
  return (
    `<ul class="autodemo-playlist">` +
    playlist
      .map((entry) => {
        const on = playing(entry.key);
        // Scenes that share a place in the loop take turns in it.
        const partners = (entry.turns_with || []).filter(playing).map((key) => playlist.find((other) => other.key === key).title);
        const note = !entry.playable
          ? `cannot play here: ${escapeHtml(entry.reason)}`
          : !on
            ? "left out"
            : partners.length
              ? `plays every other turn, taking turns with “${escapeHtml(partners.join("”, “"))}”`
              : "plays";
        return (
          `<li class="${on ? "plays" : "skipped"}"><label>` +
          `<input type="checkbox" data-scene="${escapeHtml(entry.key)}"${on ? " checked" : ""}${entry.playable ? "" : " disabled"} />` +
          `<strong>${escapeHtml(entry.title)}</strong></label><span>${note}</span></li>`
        );
      })
      .join("") +
    `</ul>`
  );
}

// The scenes that will play: listed, able to, and not taken out.
function autodemoChosen() {
  const out = new Set(AUTODEMO.options.left_out);
  return AUTODEMO.listed.filter((entry) => entry.playable && !out.has(entry.key)).map((entry) => entry.key);
}

function autodemoDrawPlaylist() {
  const box = el("autodemo-playlist");
  if (box) box.innerHTML = autodemoPlaylistHtml(AUTODEMO.listed, AUTODEMO.options.left_out);
  el("autodemo-start").disabled = !autodemoChosen().length;
}

function autodemoSceneTicked(event) {
  const tick = event.target instanceof Element ? event.target.closest("input[data-scene]") : null;
  if (!tick) return;
  const out = new Set(AUTODEMO.options.left_out);
  if (tick.checked) out.delete(tick.dataset.scene);
  else out.add(tick.dataset.scene);
  AUTODEMO.options.left_out = [...out];
  autodemoRemember();
  autodemoDrawPlaylist();
}

function autodemoRemember() {
  try {
    localStorage.setItem("ptl.autodemo", JSON.stringify(AUTODEMO.options));
  } catch {
    // Private mode: the choices just don't outlive the page.
  }
}

async function autodemoCheck() {
  const box = el("autodemo-check");
  const options = AUTODEMO.options;
  el("autodemo-start").disabled = true;
  try {
    // Asked once with the discrete GPU, to know whether there is one to leave out.
    const lang = `lang=${encodeURIComponent(options.lang)}&camera=${options.camera === "off" ? "off" : "auto"}`;
    const found = await fetchJSON(`/api/autodemo/check?dgpu=auto&${lang}`);
    const hasDiscrete = !!found.stand.dgpu;
    const hasCamera = !!found.stand.cameras_found;
    const check = options.dgpu === "off" && hasDiscrete ? await fetchJSON(`/api/autodemo/check?dgpu=off&${lang}`) : found;
    AUTODEMO.listed = check.playlist;
    const tick = (id, on, text) => `<label><input id="${id}" type="checkbox"${on ? " checked" : ""} /> ${text}</label>`;
    box.innerHTML =
      `<p class="section-label">This stand</p>${autodemoStandHtml(check.stand)}` +
      `<div class="autodemo-options">` +
      `<label>Told in <select id="autodemo-lang">` +
      [["en", "English"], ["fr", "Français"]]
        .map(([value, name]) => `<option value="${value}"${options.lang === value ? " selected" : ""}>${name}</option>`)
        .join("") +
      `</select></label>` +
      (hasDiscrete
        ? tick("autodemo-use-dgpu", options.dgpu === "auto", "Use the discrete GPU (untick it if it will be unplugged, or to rehearse a stand without it)")
        : "") +
      (hasCamera
        ? tick(
            "autodemo-use-camera",
            options.camera !== "off",
            "Use the camera: whoever is in front of it is shown on screen, and nothing is recorded (untick it in a meeting)",
          )
        : "") +
      tick("autodemo-big", options.big_screen, "Large display: bigger text, to be read from a few metres") +
      tick("autodemo-fullscreen", options.fullscreen, "Full screen") +
      `</div>` +
      `<p class="section-label">One turn of the loop: tick the demos to play</p><div id="autodemo-playlist"></div>`;
    autodemoDrawPlaylist();
    const wire = (id, change) => {
      const node = el(id);
      if (!node) return;
      node.addEventListener("change", () => {
        change(node);
        autodemoRemember();
      });
    };
    wire("autodemo-lang", (node) => {
      options.lang = node.value;
      autodemoCheck(); // the scenes' titles come in the language chosen
    });
    wire("autodemo-use-dgpu", (node) => {
      options.dgpu = node.checked ? "auto" : "off";
      autodemoCheck();
    });
    wire("autodemo-use-camera", (node) => {
      options.camera = node.checked ? "auto" : "off";
      autodemoCheck(); // a scene that showed the camera plays a video instead, or cannot play
    });
    wire("autodemo-big", (node) => (options.big_screen = node.checked));
    wire("autodemo-fullscreen", (node) => (options.fullscreen = node.checked));
  } catch (err) {
    showPlaceholder(box, `The stand could not be looked at: ${err.message}`);
  }
}

function openAutodemoModal() {
  el("autodemo-modal-overlay").hidden = false;
  el("autodemo-modal-close").focus();
  paintStatus(el("autodemo-start-status"), "", null);
  autodemoCheck();
}

function closeAutodemoModal() {
  el("autodemo-modal-overlay").hidden = true;
}

async function startAutodemo() {
  el("autodemo-start").disabled = true;
  const { fullscreen, left_out: _leftOut, ...options } = AUTODEMO.options;
  options.scenes = autodemoChosen();
  // Asked for while the click is still in hand: a browser grants the full
  // screen to a person, not to a page.
  if (fullscreen && document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
  try {
    AUTODEMO.state = await postJSON("/api/autodemo/start", options);
    closeAutodemoModal();
    location.hash = "#/stage";
  } catch (err) {
    paintStatus(el("autodemo-start-status"), `Error: ${err.message}`, "error");
    el("autodemo-start").disabled = false;
  }
}

// ------------------------------------------------------------------ the stage

const STAGE = {
  on: false,
  entered: 0,
  sceneKey: null, // the scene on the stage: its id and when it started
  view: null, // what draws its outputs
  beat: -1, // which sentence of its story is up, and since when
  beatSince: 0,
  seen: new Set(), // the stages of its demo seen at work
  figured: new Set(), // those that have had a figure to show: past loading their model
  figures: new Map(), // the last figure seen on each chip's line
  worked: new Map(), // when each chip was last seen at work
  sayTimer: null,
  popup: { open: false, left: 0, busy: false, timer: null },
};
const STAGE_POPUP_SECONDS = 15;
// A built page is shown the way a visitor would read it: from the top, then
// slowly down to the bottom. On a timer, not on animation frames, which a
// window nobody is looking at does not get.
const STAGE_PAGE_SCROLL =
  "<script>(function(){var d=document.documentElement;d.style.scrollBehavior='auto';setTimeout(function(){" +
  "var t0=Date.now(),id=setInterval(function(){var b=document.body,h=Math.max(d.scrollHeight,b?b.scrollHeight:0)-window.innerHeight," +
  "k=Math.min(1,(Date.now()-t0)/26000);window.scrollTo(0,h*k);if(k>=1)clearInterval(id);},40);},5000);})();</" +
  "script>";

function stageClock(seconds) {
  const left = Math.max(0, Math.round(seconds));
  return `${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`;
}

function stageOpen() {
  if (!STAGE.on) {
    STAGE.on = true;
    STAGE.entered = Date.now();
    STAGE.sceneKey = null;
  }
  if (AUTODEMO.state) stageRender(AUTODEMO.state);
}

function stageClose() {
  if (!STAGE.on) return;
  STAGE.on = false;
  stagePopupClose();
  stageLeaveView();
  STAGE.sceneKey = null;
  el("stage-main").replaceChildren();
  document.body.classList.remove("stage-big");
  if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {});
}

function stageLeaveView() {
  if (STAGE.view && STAGE.view.stop) STAGE.view.stop();
  STAGE.view = null;
}

function stageRender(state) {
  if (!STAGE.on || !state) return;
  document.body.classList.toggle("stage-big", !!(state.stand && state.stand.big_screen));
  const scene = state.state === "stopped" ? null : state.scene;
  const key = scene ? `${scene.id}@${scene.started_at}` : `~${state.state}`;
  if (key !== STAGE.sceneKey) {
    STAGE.sceneKey = key;
    stageLeaveView();
    STAGE.beat = -1;
    STAGE.beatSince = 0;
    STAGE.seen = new Set();
    STAGE.figured = new Set();
    STAGE.figures = new Map();
    STAGE.worked = new Map();
    const main = el("stage-main");
    main.className = `stage-main view-${scene ? scene.view : "none"}`;
    if (scene && STAGE_VIEWS[scene.view]) STAGE.view = STAGE_VIEWS[scene.view](scene, main);
    else if (state.state === "stopped") stageStoppedView(state, main);
    else main.innerHTML = `<img class="stage-waiting" src="/static/panther_lake_ai_studio_darkbg.png" alt="" />`;
  }
  stageCaption(state, scene);
  stageSide(state, scene);
  stagePill(state, scene);
  if (scene && STAGE.view && STAGE.view.update) STAGE.view.update(scene, state);
  if (STAGE.popup.open) stagePopupDraw();
}

function stageStoppedView(state, main) {
  const failures = (state.failures || [])
    .map((failure) => `<li><strong>${escapeHtml(failure.title)}</strong> ${escapeHtml(failure.error)}</li>`)
    .join("");
  main.innerHTML =
    `<div class="stage-stopped"><p>${escapeHtml(state.notice || "")}</p>` +
    (failures ? `<ul>${failures}</ul>` : "") +
    `<button type="button" class="btn btn-primary" data-stage-home>${escapeHtml(stageText().back)}</button></div>`;
}

// ---- the caption: the story, one moment at a time

function stageSay(text) {
  const node = el("stage-beat");
  if (node.dataset.text === text) return;
  const first = !node.dataset.text;
  node.dataset.text = text;
  const put = () => {
    node.textContent = text;
    stageFitBeat();
    node.classList.remove("changing");
  };
  clearTimeout(STAGE.sayTimer);
  if (first) return put();
  node.classList.add("changing");
  STAGE.sayTimer = setTimeout(put, 260);
}

// A sentence that would not fit its three lines is set smaller, not cut.
function stageFitBeat() {
  const node = el("stage-beat");
  const box = el("stage-beat-box");
  node.style.fontSize = "";
  let size = parseFloat(getComputedStyle(node).fontSize);
  while (node.offsetHeight > box.clientHeight + 1 && size > 13) {
    size -= 1;
    node.style.fontSize = `${size}px`;
  }
}

// A beat is due when everything it waits for has happened: its time into the
// scene, its stage of the demo seen at work -- and putting something out, if
// that is what the sentence points at -- the result being in.
function stageBeatDue(beat, scene, state) {
  if (beat.result && !(scene.result_ready || scene.phase === "showing")) return false;
  // One of the scene's requests, by name, has been answered.
  if (beat.answer && !(scene.answered || []).includes(beat.answer)) return false;
  if (beat.after && state.now - scene.started_at < beat.after) return false;
  if (beat.figure && !STAGE.figured.has(beat.stage)) return false;
  return !beat.stage || STAGE.seen.has(beat.stage);
}

function stageCaption(state, scene) {
  const t = stageText();
  const dots = el("stage-dots");
  if (!scene) {
    el("stage-kicker").textContent = t.tag;
    dots.replaceChildren();
    stageSay({ stopped: t.stoppedTitle, checking: t.checking, paused: t.paused }[state.state] || t.next);
    return;
  }
  // The scenes of this turn of the loop: of two that take turns, one.
  const playable = (state.playlist || []).filter((entry) => (entry.plays === undefined ? entry.playable : entry.plays));
  const position = playable.findIndex((entry) => entry.id === scene.id) + 1;
  el("stage-kicker").textContent = `${t.tag} · ${position || 1} / ${playable.length || 1} · ${scene.title}`;

  // Work of this scene only: a figure left by the last turn of the loop is
  // not this turn's stage at work.
  for (const entry of STATUS.active || []) if (entry.demo_id === scene.demo) STAGE.seen.add(entry.stage || "default");
  for (const metric of STATUS.metrics || []) {
    if (metric.demo_id !== scene.demo || metric.at < scene.started_at) continue;
    STAGE.seen.add(metric.stage || "default");
    STAGE.figured.add(metric.stage || "default");
  }

  const beats = scene.beats;
  const current = beats[STAGE.beat];
  const shown = (Date.now() - STAGE.beatSince) / 1000;
  // Long enough to be read standing up, by somebody who has just arrived.
  const reading = current ? Math.max(5, current.text.length / 24) : 0;
  let next = -1;
  // The result does not wait for the rest of the story: what is on screen
  // has changed, and the caption goes with it.
  for (let index = beats.length - 1; index > STAGE.beat && next < 0; index--) {
    if (beats[index].result && stageBeatDue(beats[index], scene, state) && shown >= Math.min(reading, 3)) next = index;
  }
  for (let index = STAGE.beat + 1; index < beats.length && next < 0 && shown >= reading; index++) {
    if (stageBeatDue(beats[index], scene, state)) next = index;
  }
  if (next >= 0) {
    STAGE.beat = next;
    STAGE.beatSince = Date.now();
    stageSay(beats[next].text);
  }
  const marks = beats.map((_beat, index) => `<i class="${index === STAGE.beat ? "now" : index < STAGE.beat ? "told" : ""}"></i>`).join("");
  if (dots.dataset.marks !== marks) {
    dots.dataset.marks = marks;
    dots.innerHTML = marks;
  }
}

// ---- the chips, down the right

function stageChipLoad(name, stand, data) {
  if (name === "CPU") return data.cpu_percent;
  if (name === "NPU") return data.npu_percent;
  const gpus = data.gpus || [];
  const id = name === "Arc Pro B60" ? stand.dgpu : stand.igpu;
  const gpu = gpus.find((entry) => entry.id === id) || (gpus.length === 1 ? gpus[0] : null);
  return gpu ? gpu.percent : null;
}

// What a chip's line shows of its work: the figure of whichever of its
// stages is at work, then the last one seen, dimmed.
function stageFigure(chip, scene, t) {
  if (!chip.demo) return { text: "", live: false, working: false };
  const stages = chip.stages || ["default"];
  // A camera clip that starts over is off its chip for a second or two:
  // its line does not blink for that.
  if ((STATUS.active || []).some((entry) => entry.demo_id === chip.demo && stages.includes(entry.stage || "default"))) {
    STAGE.worked.set(chip.chip, Date.now());
  }
  const working = Date.now() - (STAGE.worked.get(chip.chip) || 0) < 6000;
  const mine = (STATUS.metrics || []).filter(
    (metric) => metric.demo_id === chip.demo && stages.includes(metric.stage || "default") && metric.at >= scene.started_at,
  );
  const now = mine.find((metric) => !metric.sticky) || (working ? null : mine[mine.length - 1]);
  if (now) {
    STAGE.figures.set(chip.chip, formatMetric(now));
    return { text: STAGE.figures.get(chip.chip), live: working, working };
  }
  if (STAGE.figures.has(chip.chip)) return { text: STAGE.figures.get(chip.chip), live: false, working };
  return { text: working ? t.starting : "", live: false, working, note: true };
}

function stageSide(state, scene) {
  if (!STAGE.on) return;
  state = state || AUTODEMO.state;
  if (!state) return;
  if (scene === undefined) scene = state.state === "stopped" ? null : state.scene;
  const t = stageText();
  const data = STATUS.telemetry || {};
  const stand = state.stand || {};
  const names = [...(stand.npu ? ["NPU"] : []), "Integrated GPU", ...(stand.dgpu ? ["Arc Pro B60"] : []), "CPU"];
  const jobs = new Map((scene ? scene.chips : []).map((chip) => [chip.chip, chip]));
  const box = el("stage-chips");
  const shape = `${STAGE.sceneKey}|${names.join()}|${stand.lang}`;
  if (box.dataset.shape !== shape) {
    box.dataset.shape = shape;
    box.innerHTML = names
      .map((name) => {
        const job = jobs.get(name);
        const [does, model] = job ? job.label.split(" · ") : [scene ? t.idle : "", ""];
        return (
          `<div class="stage-chip${job ? " in-scene" : ""}" data-chip="${escapeHtml(name)}">` +
          `<div class="sc-head"><span class="sc-name">${escapeHtml(t.chips[name] || name)}</span><span class="sc-load">--</span></div>` +
          `<div class="telemetry-bar"><div class="telemetry-bar-fill"></div></div>` +
          `<p class="sc-does">${escapeHtml(does || "")}</p>` +
          `<p class="sc-model">${escapeHtml(model || "")}</p>` +
          `<p class="sc-figure"></p></div>`
        );
      })
      .join("");
  }
  for (const node of box.children) {
    const load = stageChipLoad(node.dataset.chip, stand, data);
    const known = load !== null && load !== undefined;
    node.querySelector(".sc-load").textContent = known ? `${Math.round(load)}%` : "";
    node.querySelector(".telemetry-bar-fill").style.width = known ? `${Math.min(load, 100)}%` : "0%";
    const job = jobs.get(node.dataset.chip);
    const figure = job && scene ? stageFigure(job, scene, t) : { text: "", live: false, working: false };
    const line = node.querySelector(".sc-figure");
    if (line.textContent !== figure.text) line.textContent = figure.text;
    line.classList.toggle("was", !figure.live && !figure.note);
    line.classList.toggle("note", !!figure.note);
    node.classList.toggle("at-work", figure.working);
  }
  const power = data.power;
  const meter = el("stage-power");
  meter.hidden = !(power && power.available);
  if (!meter.hidden) {
    el("stage-power-name").textContent = t.power;
    el("stage-power-value").textContent = `${power.package_w.toFixed(1)} W`;
    el("stage-power-fill").style.width = `${Math.min((power.package_w / POWER_SCALE_W) * 100, 100)}%`;
    const idle = power.idle_w !== null && power.idle_w !== undefined;
    el("stage-power-note").textContent = idle ? t.overIdle(Math.max(power.package_w - power.idle_w, 0).toFixed(1)) : "";
  }
}

function stagePill(state, scene) {
  const t = stageText();
  const pill = el("stage-pill");
  pill.hidden = !state.paused;
  if (!state.paused) return;
  const atWork = scene && scene.phase !== "showing";
  pill.textContent = atWork ? t.pausingPill : t.pausedPill(stageClock(state.paused.resumes_at - state.now));
}

// ---- what a demo puts out, one view per kind of demo

// The page agent: its plan, its six pictures as they are drawn, the page's
// code as it is written, then the page.
function stagePageView(scene, root) {
  const t = stageText();
  root.innerHTML =
    `<div class="sv-col">` +
    `<section class="sv-card sv-plan"><h3>${escapeHtml(t.plan)}</h3><div class="sv-plan-body">` +
    `<p class="sv-request"><span>${escapeHtml(t.request)}</span> ${escapeHtml(scene.props.request || "")}</p>` +
    `<p class="sv-wait">${escapeHtml(t.waitingPlan)}</p></div></section>` +
    `<section class="sv-card sv-pictures"><h3>${escapeHtml(t.pictures)}</h3><div class="sv-thumbs"></div></section>` +
    `</div>` +
    `<section class="sv-card sv-out"><h3>${escapeHtml(t.page)} <span class="sv-note"></span></h3>` +
    `<div class="sv-out-body"><p class="sv-wait">${escapeHtml(t.waitingPage)}</p></div></section>`;
  const planBody = root.querySelector(".sv-plan-body");
  const thumbs = root.querySelector(".sv-thumbs");
  const out = root.querySelector(".sv-out-body");
  const note = root.querySelector(".sv-note");
  let ended = false;
  let run = null; // the build this scene started, once it has been seen running
  let planShape = "";
  let code = null;
  let frame = null;
  let framed = false;
  let ready = false;

  const drawPlan = (state) => {
    const plan = state.plan;
    if (!plan) return;
    const shape = JSON.stringify([plan.title, plan.headline, plan.offers, plan.pictures.map((picture) => [picture.name, picture.ready])]);
    if (shape === planShape) return;
    planShape = shape;
    const offers = plan.offers || [];
    planBody.innerHTML =
      `<p class="sv-request"><span>${escapeHtml(t.request)}</span> ${escapeHtml(scene.props.request || "")}</p>` +
      (plan.title ? `<p class="sv-plan-title">${escapeHtml(plan.title)}</p>` : "") +
      (plan.headline ? `<p class="sv-plan-headline">${escapeHtml(plan.headline)}</p>` : "") +
      (offers.length ? `<p class="sv-plan-offers"><span>${escapeHtml(t.onOffer)}</span> ${offers.map(escapeHtml).join(" · ")}</p>` : "");
    // Six frames from the start, filled as each picture is drawn; a picture
    // already up is left alone, so that it does not blink at every look.
    const names = plan.pictures.slice(0, 6).map((picture) => picture.name);
    if (thumbs.dataset.names !== names.join()) {
      thumbs.dataset.names = names.join();
      thumbs.innerHTML = names.map((name) => `<div class="sv-thumb" data-name="${escapeHtml(name)}"></div>`).join("");
    }
    for (const picture of plan.pictures.slice(0, 6)) {
      const tile = [...thumbs.children].find((node) => node.dataset.name === picture.name);
      if (!tile || !picture.ready || tile.firstChild) continue;
      const image = document.createElement("img");
      image.alt = "";
      image.src = `/api/page-agent/picture/${encodeURIComponent(picture.name)}?run=${state.run}`;
      tile.append(image);
    }
  };

  const progress = async () => {
    try {
      const state = await fetchJSON("/api/page-agent/progress");
      if (ended) return;
      // The launcher keeps the last build's progress until the next one
      // starts: only this scene's own build is drawn.
      if (state.running && run === null) run = state.run;
      if (state.run === run || (run === null && ready)) drawPlan(state);
    } catch {
      // A missed look: the next one has everything.
    }
  };

  const partial = async () => {
    if (framed) return;
    try {
      const data = await fetchJSON("/api/bricks/page-agent/partial?stage=page");
      if (ended || framed || !data.active || !data.text) return;
      if (!code) {
        code = document.createElement("pre");
        code.className = "sv-code";
        out.replaceChildren(code);
        note.textContent = t.writing;
      }
      // The end of it: what is being written now.
      code.textContent = data.text.slice(-5000);
      code.scrollTop = code.scrollHeight;
    } catch {
      // A missed look.
    }
  };

  const fit = () => {
    if (!frame) return;
    const holder = frame.parentElement;
    const scale = holder.clientWidth / 1280;
    frame.style.width = "1280px";
    frame.style.height = `${Math.ceil(holder.clientHeight / scale)}px`;
    frame.style.transform = `scale(${scale})`;
  };

  const showPage = async () => {
    try {
      const result = await fetchJSON("/api/autodemo/result");
      if (ended || result.scene !== scene.id || !result.data || !result.data.html) return;
      const holder = document.createElement("div");
      holder.className = "sv-frame";
      frame = document.createElement("iframe");
      frame.setAttribute("sandbox", "allow-scripts");
      frame.title = "The page that was built";
      frame.srcdoc = result.data.html + STAGE_PAGE_SCROLL;
      holder.append(frame);
      out.replaceChildren(holder);
      note.textContent = t.built;
      fit();
      progress(); // the last pictures, if the page was quicker than the look that draws them
    } catch {
      framed = false; // tried again at the next look
    }
  };

  const timers = [setInterval(progress, 1000), setInterval(partial, 400)];
  window.addEventListener("resize", fit);
  progress();
  return {
    update(now) {
      ready = !!now.result_ready;
      if (ready && !framed) {
        framed = true;
        showPage();
      }
    },
    stop() {
      ended = true;
      timers.forEach(clearInterval);
      window.removeEventListener("resize", fit);
    },
  };
}

// The receipts: each one beside the line the two models make of it.
function stageReceiptsView(scene, root) {
  const t = stageText();
  const receipts = scene.props.receipts || [];
  root.innerHTML =
    `<div class="sv-receipts${receipts.length > 3 ? " many" : ""}">` +
    receipts
      .map(
        (receipt) =>
          `<article class="sv-card sv-receipt" data-file="${escapeHtml(receipt.name)}">` +
          `<div class="sv-receipt-picture"><img src="${escapeHtml(receipt.url)}" alt="" /></div>` +
          `<div class="sv-receipt-line"><p class="sv-wait">${escapeHtml(t.receiptWaiting)}</p></div></article>`,
      )
      .join("") +
    `</div><p class="sv-total"></p>`;
  const cards = [...root.querySelectorAll(".sv-receipt")];
  const total = root.querySelector(".sv-total");
  let ended = false;
  const money = (amount, currency) => `${Number(amount).toFixed(2)} ${currency || ""}`.trim();
  // What the brick itself is not sure of: a figure it could not find on the
  // receipt, a field the model left out.
  const doubts = (item) => [item.error, ...(item.review_reasons || [])].filter(Boolean);

  const look = async () => {
    try {
      const report = await fetchJSON("/api/expense-extract/report");
      if (ended) return;
      const items = report.items || [];
      let reading = !!report.running; // the first receipt without a line is the one being read
      for (const card of cards) {
        const item = items.find((entry) => String(entry.source_file || "").split(/[\\/]/).pop() === card.dataset.file);
        const line = card.querySelector(".sv-receipt-line");
        const shape = item ? `item:${item.id}:${item.revision}` : reading ? "reading" : "waiting";
        card.classList.toggle("reading", shape === "reading");
        card.classList.toggle("done", !!item);
        card.classList.toggle("flagged", !!item && doubts(item).length > 0);
        if (!item) reading = false;
        if (line.dataset.shape === shape) continue;
        line.dataset.shape = shape;
        if (!item) {
          line.innerHTML = `<p class="sv-wait">${escapeHtml(shape === "reading" ? t.receiptReading : t.receiptWaiting)}</p>`;
          continue;
        }
        const row = (label, value, unsure = false) =>
          `<dt>${escapeHtml(label)}</dt><dd${unsure ? ' class="unsure"' : ""}>` +
          `${escapeHtml(value === null || value === undefined ? "" : String(value))}</dd>`;
        // A figure the brick could not find on the receipt is shown as what it is.
        const unsure = (field) => doubts(item).some((reason) => reason.startsWith(field));
        line.innerHTML =
          `<dl>${row(t.vendor, item.vendor, unsure("Vendor"))}${row(t.date, item.date, unsure("Date"))}` +
          `${row(t.amount, item.amount === null || item.amount === undefined ? "" : money(item.amount, item.currency), unsure("Amount") || unsure("Currency"))}` +
          `${row(t.category, item.category, unsure("Category"))}</dl>` +
          doubts(item)
            .slice(0, 1)
            .map((reason) => `<p class="sv-flag">${escapeHtml(t.toCheck)}: ${escapeHtml(t.reasons[reason] || reason.toLowerCase())}</p>`)
            .join("");
      }
      if (!report.running && items.length) {
        // As the brick's own totals do: a line somebody has to check is not
        // counted yet, and currencies are never added together.
        const sums = new Map();
        const counted = items.filter((item) => !doubts(item).length && item.amount !== null && item.amount !== undefined);
        for (const item of counted) sums.set(item.currency || "", (sums.get(item.currency || "") || 0) + Number(item.amount));
        total.textContent = t.receiptsDone(
          items.length,
          [...sums].map(([currency, sum]) => money(sum, currency)).join(" + "),
          items.length - counted.length,
        );
      }
    } catch {
      // A missed look.
    }
  };

  const timer = setInterval(look, 1200);
  look();
  return {
    stop() {
      ended = true;
      clearInterval(timer);
    },
  };
}

// The street cameras: each one's picture with what was found drawn on it,
// the chip that watches it, how fast, and what it has counted.
function stageCamerasView(scene, root) {
  const t = stageText();
  const feeds = scene.props.feeds || [];
  root.innerHTML =
    `<div class="sv-cameras">` +
    feeds
      .map(
        (feed) =>
          `<article class="sv-card sv-camera" data-feed="${escapeHtml(feed.id)}">` +
          `<div class="sv-camera-head"><strong>${escapeHtml(feed.name)}</strong>` +
          `<span class="sv-camera-chip">${escapeHtml(t.chips[feed.chip] || feed.chip)}</span><span class="sv-camera-rate"></span></div>` +
          `<div class="sv-camera-picture"><p class="sv-wait">${escapeHtml(t.connecting)}</p></div>` +
          `<p class="sv-camera-counts"><span>${escapeHtml(t.counted)}</span> <b>${escapeHtml(t.nothingYet)}</b></p></article>`,
      )
      .join("") +
    `</div>`;
  const cards = [...root.querySelectorAll(".sv-camera")];
  let ended = false;

  const look = async () => {
    try {
      const data = await fetchJSON("/api/smart-city-monitor/counts");
      if (ended) return;
      const totals = (data.snapshot && data.snapshot.per_feed_total) || {};
      for (const card of cards) {
        const id = card.dataset.feed;
        const picture = card.querySelector(".sv-camera-picture");
        if (data.running && !picture.querySelector("img")) {
          const image = document.createElement("img");
          image.alt = "";
          // The stream ends if the camera is not up yet: asked again at the next look.
          image.addEventListener("error", () => image.remove());
          image.addEventListener("load", () => picture.querySelector(".sv-wait")?.remove(), { once: true });
          image.src = `/api/smart-city-monitor/stream?feed=${encodeURIComponent(id)}&at=${Date.now()}`;
          picture.append(image);
        }
        const counted = Object.entries(totals[id] || {})
          .filter(([, count]) => count > 0)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 4)
          .map(([kind, count]) => `${kind} ${count}`)
          .join(" · ");
        card.querySelector(".sv-camera-counts b").textContent = counted || t.nothingYet;
        const rate = (STATUS.metrics || []).find((metric) => metric.demo_id === scene.demo && metric.stage === id && !metric.sticky);
        card.querySelector(".sv-camera-rate").textContent = rate ? formatMetric(rate) : "";
      }
    } catch {
      // A missed look.
    }
  };

  const timer = setInterval(look, 1000);
  look();
  return {
    stop() {
      ended = true;
      clearInterval(timer);
      // An <img> on a stream holds its connection open until told otherwise.
      for (const image of root.querySelectorAll(".sv-camera-picture img")) image.src = "";
    },
  };
}

// The commentator: the video with its line on it as a subtitle, what the
// vision model actually saw underneath, and the lines said so far, each with
// the voice it was said in.
function stageCommentaryView(scene, root) {
  const t = stageText();
  const moods = scene.props.moods || {};
  root.innerHTML =
    `<article class="sv-card sv-camera sv-watch">` +
    `<div class="sv-camera-head"><strong>${escapeHtml(scene.props.video || "")}</strong>` +
    `<span class="sv-camera-chip" hidden></span><span class="sv-camera-rate"></span></div>` +
    `<div class="sv-camera-picture"><p class="sv-wait">${escapeHtml(t.loadingModels)}</p><p class="sv-subtitle" hidden></p></div>` +
    `<p class="sv-camera-counts"><span>${escapeHtml(t.saw)}</span> <b>${escapeHtml(t.firstLine)}</b></p></article>` +
    `<section class="sv-card sv-said"><h3>${escapeHtml(t.saidTitle)}</h3><div class="sv-lines"><p class="sv-wait">${escapeHtml(t.firstLine)}</p></div></section>`;
  const picture = root.querySelector(".sv-camera-picture");
  const subtitle = root.querySelector(".sv-subtitle");
  const badge = root.querySelector(".sv-camera-chip");
  const lines = root.querySelector(".sv-lines");
  let ended = false;
  let last = 0;

  const look = async () => {
    try {
      const data = await fetchJSON(`/api/video-commentary/comments?after=${last}`);
      if (ended) return;
      if (data.running && !picture.querySelector("img")) {
        const image = document.createElement("img");
        image.alt = "";
        // The stream ends if the video is not up yet: asked again at the next look.
        image.addEventListener("error", () => image.remove());
        image.addEventListener("load", () => picture.querySelector(".sv-wait")?.remove(), { once: true });
        image.src = `/api/video-commentary/stream?at=${Date.now()}`;
        picture.prepend(image);
      }
      for (const comment of data.comments || []) {
        if (!last) lines.replaceChildren();
        last = comment.number;
        const plain = comment.said === comment.seen;
        const voice = moods[comment.mood] || comment.mood;
        subtitle.textContent = comment.said;
        subtitle.hidden = false;
        badge.textContent = voice;
        badge.hidden = false;
        root.querySelector(".sv-camera-counts b").textContent = comment.seen;
        root.querySelector(".sv-camera-rate").textContent =
          t.toSee(comment.seeing_seconds.toFixed(1)) + (plain ? "" : ` · ${t.toSay(comment.saying_seconds.toFixed(1))}`);
        const entry = document.createElement("div");
        entry.className = "sv-line";
        entry.innerHTML = `<span>${escapeHtml(voice)}</span><p>${escapeHtml(comment.said)}</p>`;
        lines.prepend(entry);
        while (lines.children.length > 6) lines.lastChild.remove();
      }
    } catch {
      // A missed look.
    }
  };

  const timer = setInterval(look, 700);
  look();
  return {
    stop() {
      ended = true;
      clearInterval(timer);
      // An <img> on a stream holds its connection open until told otherwise.
      for (const image of picture.querySelectorAll("img")) image.src = "";
    },
  };
}

// The documents: the folder's files down the left, and the same question
// asked twice down the right -- of the model alone, then with the files
// read. What changes between the two is what the scene is about.
function stageDocumentsView(scene, root) {
  const t = stageText();
  const files = scene.props.files || [];
  const ask = (which, title) =>
    `<div class="sv-ask" data-ask="${which}"><h3>${escapeHtml(title)} <span class="sv-note"></span></h3>` +
    `<p class="sv-question">${escapeHtml(scene.props.question || "")}</p>` +
    `<div class="sv-answer"><p class="sv-wait"></p></div></div>`;
  root.innerHTML =
    `<section class="sv-card sv-folder"><h3>${escapeHtml(t.folder)} <span class="sv-note">${escapeHtml(scene.props.folder || "")}</span></h3>` +
    `<ul class="sv-files">` +
    files.map((name) => `<li data-file="${escapeHtml(name)}"><span class="sv-file-name">${escapeHtml(name)}</span><span class="sv-file-state"></span></li>`).join("") +
    `</ul><p class="sv-folder-note">${escapeHtml(t.notShown)}</p></section>` +
    `<section class="sv-card sv-asks">${ask("alone", t.askedAlone)}<p class="sv-read"></p>${ask("with", t.askedWith)}</section>`;
  const block = (which) => root.querySelector(`.sv-ask[data-ask="${which}"]`);
  const rows = [...root.querySelectorAll(".sv-files li")];
  let ended = false;
  let fetching = false;
  const shown = new Set(); // the answers that are up, whole

  const put = (which, words, note) => {
    const box = block(which).querySelector(".sv-answer");
    if (!box.querySelector(".sv-answer-text")) box.innerHTML = `<p class="sv-answer-text"></p>`;
    box.querySelector(".sv-answer-text").textContent = words;
    block(which).querySelector(".sv-note").textContent = note;
    block(which).classList.add("on");
  };
  const wait = (which, words) => {
    const node = block(which).querySelector(".sv-wait");
    if (node) node.textContent = words;
  };
  const mark = (state, only) => {
    for (const row of rows) {
      const on = !only || only.has(row.dataset.file);
      row.className = on ? state : "read";
      row.querySelector(".sv-file-state").textContent = on ? t.fileStates[state] : t.fileStates.read;
    }
  };
  mark("out");
  wait("alone", t.asking);
  wait("with", t.notYet);

  // The answer as it is written, in whichever of the two is being asked.
  const writing = async () => {
    try {
      const data = await fetchJSON("/api/bricks/doc-qa/partial");
      if (ended || !data.active || !data.text) return;
      const which = shown.has("alone") ? "with" : "alone";
      if (which === "with" && !shown.has("read")) return;
      if (!shown.has(which)) put(which, data.text, t.writing);
    } catch {
      // A missed look.
    }
  };

  const collect = async (answered) => {
    if (fetching) return;
    fetching = true;
    try {
      const result = await fetchJSON("/api/autodemo/result");
      if (ended || result.scene !== scene.id) return;
      const answers = result.answers || {};
      if (answered.includes("alone") && !shown.has("alone") && answers.alone) {
        shown.add("alone");
        put("alone", answers.alone.text, t.noFiles);
      }
      if (answered.includes("read") && !shown.has("read") && answers.read) {
        shown.add("read");
        mark("read");
        root.querySelector(".sv-read").textContent = t.filesRead((answers.read.files || files).length, answers.read.chunks);
        root.querySelector(".sv-folder-note").textContent = t.nowRead;
        wait("with", t.asking);
      }
      if (answered.includes("with") && !shown.has("with") && answers.with) {
        shown.add("with");
        put("with", answers.with.text, t.fromFiles);
        const used = new Set((answers.with.sources || []).map((source) => String(source.source).split(/[\\/]/).pop()));
        if (used.size) mark("used", used);
      }
    } catch {
      // Tried again at the next look.
    } finally {
      fetching = false;
    }
  };

  const timer = setInterval(writing, 400);
  return {
    update(now) {
      const answered = now.answered || [];
      if (answered.some((name) => !shown.has(name))) collect(answered);
    },
    stop() {
      ended = true;
      clearInterval(timer);
    },
  };
}

// The camera: whoever is in front of it, boxed by the detector, with what the
// vision model makes of the picture on it as a subtitle; what is in the
// picture now, how fast, and the sentences so far.
function stageCameraView(scene, root) {
  const t = stageText();
  root.innerHTML =
    `<article class="sv-card sv-camera sv-watch">` +
    `<div class="sv-camera-head"><strong>${escapeHtml(t.camera)}</strong>` +
    `<span class="sv-camera-chip">${escapeHtml(t.chips[scene.props.chip] || scene.props.chip || "")}</span><span class="sv-camera-rate"></span></div>` +
    `<div class="sv-camera-picture"><p class="sv-wait">${escapeHtml(t.startingDetector)}</p><p class="sv-subtitle" hidden></p></div>` +
    `<p class="sv-camera-counts"><span>${escapeHtml(t.inPicture)}</span> <b>${escapeHtml(t.nothingSeen)}</b></p></article>` +
    `<section class="sv-card sv-said"><h3>${escapeHtml(t.saidOfIt)}</h3><div class="sv-lines"><p class="sv-wait">${escapeHtml(t.firstLine)}</p></div></section>`;
  const picture = root.querySelector(".sv-camera-picture");
  const subtitle = root.querySelector(".sv-subtitle");
  const lines = root.querySelector(".sv-lines");
  let ended = false;
  let last = 0;

  const watch = async () => {
    try {
      const data = await fetchJSON("/api/object-detection/detections");
      if (ended) return;
      if (data.watching && !picture.querySelector("img")) {
        const image = document.createElement("img");
        image.alt = "";
        // The stream ends if the camera is not up yet: asked again at the next look.
        image.addEventListener("error", () => image.remove());
        image.addEventListener("load", () => picture.querySelector(".sv-wait")?.remove(), { once: true });
        image.src = `/api/object-detection/stream?at=${Date.now()}`;
        picture.prepend(image);
      }
      const seen = Object.entries(data.counts || {})
        .slice(0, 5)
        .map(([kind, count]) => `${kind.replace(/_/g, " ")} ${count}`)
        .join(" · ");
      root.querySelector(".sv-camera-counts b").textContent = seen || t.nothingSeen;
      const rate = (STATUS.metrics || []).find((metric) => metric.demo_id === "object-detection" && !metric.sticky);
      root.querySelector(".sv-camera-rate").textContent = rate ? formatMetric(rate) : "";
    } catch {
      // A missed look.
    }
  };

  const listen = async () => {
    try {
      const data = await fetchJSON(`/api/video-commentary/comments?after=${last}`);
      if (ended) return;
      for (const comment of data.comments || []) {
        if (!last) lines.replaceChildren();
        last = comment.number;
        subtitle.textContent = comment.said;
        subtitle.hidden = false;
        const entry = document.createElement("div");
        entry.className = "sv-line";
        entry.innerHTML = `<span>${escapeHtml(t.toSee(comment.seeing_seconds.toFixed(1)))}</span><p>${escapeHtml(comment.said)}</p>`;
        lines.prepend(entry);
        while (lines.children.length > 6) lines.lastChild.remove();
      }
    } catch {
      // A missed look.
    }
  };

  const timers = [setInterval(watch, 1000), setInterval(listen, 700)];
  watch();
  return {
    stop() {
      ended = true;
      timers.forEach(clearInterval);
      // An <img> on a stream holds its connection open until told otherwise.
      for (const image of picture.querySelectorAll("img")) image.src = "";
    },
  };
}

const STAGE_VIEWS = {
  page: stagePageView,
  receipts: stageReceiptsView,
  cameras: stageCamerasView,
  commentary: stageCommentaryView,
  documents: stageDocumentsView,
  camera: stageCameraView,
};

// ---- somebody at the machine

function stagePopupDraw() {
  const t = stageText();
  const popup = STAGE.popup;
  const paused = !!(AUTODEMO.state && AUTODEMO.state.paused);
  const box = el("stage-popup");
  const count = ` (${popup.left})`;
  el("stage-popup-title").textContent = t.popupTitle;
  el("stage-popup-text").textContent = popup.busy ? t.stopping : paused ? t.popupPaused : t.popupText;
  // The count sits on what happens if nobody answers: a stand that waits
  // for an answer from somebody who has walked away has stopped for the day.
  box.querySelector('[data-stage-action="continue"]').textContent = paused ? t.resume : t.keep + count;
  box.querySelector('[data-stage-action="pause"]').textContent = paused ? t.stay + count : t.pause;
  box.querySelector('[data-stage-action="stop"]').textContent = t.stop;
  box.querySelector('[data-stage-action="skip"]').textContent = t.skip;
  for (const button of box.querySelectorAll("button")) button.disabled = popup.busy;
}

function stagePopupOpen() {
  if (STAGE.popup.open || !autodemoActive()) return;
  const paused = !!AUTODEMO.state.paused;
  STAGE.popup = { open: true, left: STAGE_POPUP_SECONDS, busy: false, timer: null };
  // Somebody is here: a pause is not about to end under their hands.
  if (paused) postJSON("/api/autodemo/pause", {}).catch(() => {});
  stagePopupDraw();
  el("stage-popup").hidden = false;
  el("stage-popup").querySelector(`[data-stage-action="${paused ? "pause" : "continue"}"]`).focus();
  STAGE.popup.timer = setInterval(() => {
    if (STAGE.popup.busy) return;
    STAGE.popup.left -= 1;
    if (STAGE.popup.left <= 0) stagePopupAct("unanswered");
    else stagePopupDraw();
  }, 1000);
}

function stagePopupClose() {
  clearInterval(STAGE.popup.timer);
  STAGE.popup = { open: false, left: 0, busy: false, timer: null };
  el("stage-popup").hidden = true;
  STAGE.entered = Date.now(); // the click that answered is not another visitor
}

async function stagePopupAct(action) {
  if (STAGE.popup.busy) return;
  const paused = !!(AUTODEMO.state && AUTODEMO.state.paused);
  if (action === "unanswered") action = paused ? "pause" : "continue";
  try {
    if (action === "stop") {
      STAGE.popup.busy = true;
      stagePopupDraw();
      AUTODEMO.state = await postJSON("/api/autodemo/stop", {});
      stagePopupClose();
      location.hash = "#/";
      return;
    }
    if (action === "continue" && paused) AUTODEMO.state = await postJSON("/api/autodemo/resume", {});
    if (action === "pause") AUTODEMO.state = await postJSON("/api/autodemo/pause", {});
    if (action === "skip") AUTODEMO.state = await postJSON("/api/autodemo/skip", {});
  } catch {
    // The launcher did not answer: the next look says where the loop is.
  }
  stagePopupClose();
  stageRender(AUTODEMO.state);
}

// A press or a key, not a mouse that moved: a stand gets bumped.
function stageTouched(event) {
  if (!STAGE.on || !autodemoActive()) return;
  if (event.type === "keydown") {
    if (["Shift", "Control", "Alt", "Meta", "F11"].includes(event.key)) return;
    if (STAGE.popup.open) {
      if (event.key === "Escape") stagePopupAct("unanswered");
      return;
    }
  } else if (STAGE.popup.open) {
    // Beside the card: the same as not answering.
    if (event.target === el("stage-popup")) stagePopupAct("unanswered");
    return;
  }
  if (Date.now() - STAGE.entered < 1500) return; // the click that started the loop, or answered the popup
  stagePopupOpen();
}

// ---- following the loop

async function autodemoTick() {
  let state;
  try {
    state = await fetchJSON("/api/autodemo");
  } catch {
    return; // the launcher is restarting or away: the next look will tell
  }
  AUTODEMO.state = state;
  const onStage = location.hash === "#/stage";
  // The loop runs on this machine whatever the page shows: a page opened or
  // reloaded while it does joins it, and one left on the stage when it has
  // ended goes back to the start.
  if (autodemoActive() && !onStage) {
    location.hash = "#/stage";
    return;
  }
  if (onStage && !autodemoActive() && state.state !== "stopped") {
    location.hash = "#/";
    return;
  }
  if (onStage) stageRender(state);
  else if (!el("landing").hidden) landingRender();
}

function wireAutodemo() {
  try {
    Object.assign(AUTODEMO.options, JSON.parse(localStorage.getItem("ptl.autodemo") || "{}"));
  } catch {
    // Nothing remembered.
  }
  el("landing-auto").addEventListener("click", openAutodemoModal);
  el("autodemo-modal-close").addEventListener("click", closeAutodemoModal);
  el("autodemo-modal-overlay").addEventListener("click", (event) => {
    if (event.target === el("autodemo-modal-overlay")) closeAutodemoModal();
  });
  el("autodemo-start").addEventListener("click", startAutodemo);
  el("autodemo-check").addEventListener("change", autodemoSceneTicked);
  el("stage").addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (target && target.closest("[data-stage-home]")) location.hash = "#/";
    const action = target && target.closest("[data-stage-action]");
    if (action) stagePopupAct(action.dataset.stageAction);
  });
  el("stage").addEventListener("pointerdown", stageTouched);
  document.addEventListener("keydown", stageTouched);
  setInterval(autodemoTick, AUTODEMO_POLL_MS);
  return autodemoTick();
}

// --- Updates ----------------------------------------------------------------------
// The launcher asks GitHub (through git) whether a newer version exists as it
// starts. The page offers it once per launcher start -- the server remembers,
// so a reload doesn't ask again -- and the footer keeps an Upgrade button.
// Upgrading hands off to a helper that stops the launcher, pulls, updates the
// dependencies and starts it again; this page waits for that, then reloads.

let UPDATE = null;
let upgrading = false;

const waitMs = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function loadUpdateStatus({ prompt = false } = {}) {
  // The check is a git fetch started with the launcher: wait for its answer
  // rather than saying "up to date" before there is one.
  for (let attempt = 0; attempt < 45; attempt++) {
    try {
      UPDATE = await fetchJSON("/api/update");
    } catch {
      return;
    }
    if (UPDATE.checked_at && !UPDATE.checking) break;
    await waitMs(1000);
  }
  renderUpdateFooter();
  if (prompt && UPDATE.prompt) {
    openUpdateModal();
    postJSON("/api/update/prompted", {}).catch(() => {});
  }
}

function renderUpdateFooter() {
  const state = el("update-state");
  const button = el("update-open");
  if (!UPDATE || !UPDATE.checked_at) {
    state.hidden = true;
    button.hidden = true;
    return;
  }
  button.hidden = !UPDATE.update_available;
  state.hidden = UPDATE.update_available;
  if (UPDATE.update_available) {
    button.textContent = `Upgrade to v${UPDATE.latest}`;
    button.title = UPDATE.can_upgrade ? "Install the new version and restart the launcher" : UPDATE.blocked_reason || "";
  } else if (UPDATE.error) {
    state.textContent = "Couldn't check for updates -- retry";
    state.title = UPDATE.error;
  } else {
    state.textContent = "Up to date -- check again";
    state.title = `Checked against GitHub at ${new Date(UPDATE.checked_at * 1000).toLocaleTimeString()}`;
  }
}

async function recheckForUpdates() {
  const state = el("update-state");
  state.textContent = "Checking GitHub…";
  try {
    UPDATE = await postJSON("/api/update/check", {});
  } catch (err) {
    state.textContent = "Couldn't check for updates -- retry";
    state.title = err.message;
    return;
  }
  renderUpdateFooter();
}

async function openUpdateFromFooter() {
  try {
    // Fresh: which demos are running decides whether Upgrade now is allowed.
    UPDATE = await fetchJSON("/api/update");
  } catch {
    // Fall back to what the footer already knew.
  }
  openUpdateModal();
}

function setUpdateNotice(text, { link = false } = {}) {
  const notice = el("update-notice");
  notice.hidden = !text;
  notice.innerHTML = text
    ? escapeHtml(text) + (link ? ` <a href="${escapeHtml(UPDATE.repo_url)}" target="_blank" rel="noopener">Open on GitHub</a>` : "")
    : "";
}

function setUpgradeProgress(text) {
  el("update-progress").hidden = !text;
  el("update-progress-text").textContent = text || "";
}

// "2026-10-03" the way the reader's own locale writes a day.
function formatDay(isoDay) {
  const day = new Date(`${isoDay}T00:00:00`);
  if (Number.isNaN(day.getTime())) return isoDay;
  return day.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

// What changed, under the version that shipped it, newest first. A change
// with more to say (its commit message's opening paragraph) opens on click.
// `current`, if given, is the version to mark as the one running.
function changelogHtml(versions, heading, { current = null } = {}) {
  const blocks = (versions || []).filter((version) => (version.changes || []).length);
  if (!blocks.length) return "";
  const count = blocks.filter((version) => version.version).length;
  const body = blocks
    .map((version) => {
      const title =
        escapeHtml(version.version ? `v${version.version}` : "Not numbered yet") +
        (current && version.version === current ? '<span class="changelog-tag">this version</span>' : "") +
        (version.date ? `<span class="changelog-date">${escapeHtml(formatDay(version.date))}</span>` : "");
      const items = version.changes
        .map((change) =>
          change.details
            ? `<li><details><summary>${escapeHtml(change.summary)}</summary><p>${escapeHtml(change.details)}</p></details></li>`
            : `<li>${escapeHtml(change.summary)}</li>`,
        )
        .join("");
      return `<section class="changelog-version"><h3>${title}</h3><ul class="update-change-list">${items}</ul></section>`;
    })
    .join("");
  const label = count > 1 ? `${heading} -- ${count} versions` : heading;
  return `<p class="section-label">${escapeHtml(label)}</p><div class="changelog" tabindex="0" role="region" aria-label="${escapeHtml(heading)}">${body}</div>`;
}

function openUpdateModal() {
  if (!UPDATE || !UPDATE.update_available) return;
  el("update-modal-title").textContent = "Update available";
  el("update-summary").textContent = `Panther Lake AI Studio v${UPDATE.latest} is available -- this copy runs v${UPDATE.local}.`;
  // Each version this upgrade would jump through, with what it changed. A
  // copy that can't ask git (a zip download) has no list to show.
  el("update-changes").innerHTML = changelogHtml(UPDATE.changelog, "What's new");
  const running = UPDATE.running_demos || [];
  if (!UPDATE.can_upgrade) {
    setUpdateNotice(UPDATE.blocked_reason || "This copy can't upgrade itself.", { link: true });
  } else if (running.length) {
    setUpdateNotice(`Stop the running demos first (${running.join(", ")}): upgrading restarts the launcher.`);
  } else {
    setUpdateNotice(null);
  }
  const blocked = !UPDATE.can_upgrade || running.length > 0;
  el("update-explainer").hidden = false;
  el("update-now").hidden = false;
  el("update-now").disabled = blocked;
  el("update-later").hidden = false;
  el("update-later").textContent = "Later";
  el("update-modal-close").hidden = false;
  setUpgradeProgress(null);
  el("update-modal-overlay").hidden = false;
  (blocked ? el("update-later") : el("update-now")).focus();
}

function closeUpdateModal() {
  if (upgrading) return; // the page is about to reload on its own
  el("update-modal-overlay").hidden = true;
}

async function startUpgrade() {
  const target = UPDATE.latest;
  const from = UPDATE.local;
  upgrading = true;
  el("update-now").disabled = true;
  el("update-later").hidden = true;
  el("update-modal-close").hidden = true;
  setUpdateNotice(null);
  setUpgradeProgress("Starting the upgrade…");
  try {
    await postJSON("/api/update/upgrade", {});
  } catch (err) {
    upgrading = false;
    el("update-now").disabled = false;
    el("update-later").hidden = false;
    el("update-modal-close").hidden = false;
    setUpgradeProgress(null);
    setUpdateNotice(err.message);
    return;
  }
  el("update-modal-title").textContent = `Upgrading to v${target}`;
  el("update-explainer").hidden = true;
  const started = Date.now();
  let wentDown = false;
  // Down while the helper works, then up again -- on the new version, or on
  // the old one if a step failed; the page after the reload says which.
  while (Date.now() - started < 15 * 60 * 1000) {
    const seconds = Math.round((Date.now() - started) / 1000);
    try {
      const running = await fetchJSON("/api/version");
      if (wentDown || running.version !== from) {
        location.href = `${location.pathname}?upgraded=1${location.hash}`;
        return;
      }
      setUpgradeProgress(`Stopping the launcher… ${seconds} s`);
    } catch {
      wentDown = true;
      setUpgradeProgress(`Installing v${target} and restarting -- this page reloads by itself. ${seconds} s`);
    }
    await waitMs(1500);
  }
  setUpgradeProgress("This is taking longer than expected: see logs/upgrade.log in the app folder.");
}

// After the reload an upgrade triggers: say how it went, once.
async function showUpgradeResult() {
  if (!new URLSearchParams(location.search).has("upgraded")) return false;
  history.replaceState(null, "", location.pathname + location.hash);
  let result = null;
  try {
    result = await fetchJSON("/api/update/last");
  } catch {
    return false;
  }
  if (!result) return false;
  postJSON("/api/update/prompted", {}).catch(() => {});
  el("update-modal-title").textContent = result.ok ? `Upgraded to v${result.to}` : "The upgrade didn't complete";
  el("update-summary").textContent = result.ok
    ? `Panther Lake AI Studio now runs v${result.to} (it was v${result.from}).`
    : `It stopped at the ${result.step} step; the launcher started again on v${result.to || result.from}. Full log: logs/upgrade.log.`;
  el("update-changes").innerHTML = result.ok
    ? changelogHtml(result.changelog, "What changed")
    : `<pre class="text-block update-log">${escapeHtml(result.error || "")}</pre>`;
  el("update-explainer").hidden = true;
  setUpdateNotice(null);
  setUpgradeProgress(null);
  el("update-now").hidden = true;
  el("update-later").hidden = false;
  el("update-later").textContent = "Close";
  el("update-modal-close").hidden = false;
  el("update-modal-overlay").hidden = false;
  el("update-later").focus();
  return true;
}

// --- Version history ----------------------------------------------------------------
// The version in the footer opens every version this copy has been through and
// what each changed. Read from the copy's own git history: no network needed.

async function openChangelogModal() {
  el("changelog-summary").textContent = "Reading this copy's history…";
  el("changelog-notice").hidden = true;
  el("changelog-body").innerHTML = "";
  el("changelog-modal-overlay").hidden = false;
  el("changelog-modal-close").focus();
  let data;
  try {
    data = await fetchJSON("/api/changelog");
  } catch (err) {
    el("changelog-summary").textContent = `Couldn't read the version history: ${err.message}`;
    return;
  }
  let summary = `This copy runs v${data.running}.`;
  if (data.on_disk && data.on_disk !== data.running) {
    summary += ` v${data.on_disk} is on disk: restart the launcher to run it.`;
  }
  if (UPDATE && UPDATE.update_available) {
    summary += ` v${UPDATE.latest} is available -- see Upgrade in the footer.`;
  }
  el("changelog-summary").textContent = summary;
  if (!data.available) {
    const notice = el("changelog-notice");
    notice.hidden = false;
    notice.innerHTML =
      `${escapeHtml(data.reason || "This copy's history couldn't be read.")} ` +
      `<a href="${escapeHtml(data.repo_url)}/commits/main" target="_blank" rel="noopener">Open on GitHub</a>`;
    return;
  }
  el("changelog-body").innerHTML =
    changelogHtml(data.versions, "Every version, newest first", { current: data.running }) ||
    '<p class="placeholder">No changes are recorded in this copy\'s history.</p>';
}

function closeChangelogModal() {
  el("changelog-modal-overlay").hidden = true;
  el("app-version").focus();
}

// --- Models -----------------------------------------------------------------------
// What each demo loads, whether this machine has it, and fetching what's missing
// before a show instead of in front of an audience (BACKLOG R10). Sizes come from
// the Hub; "do we have it" is answered from the cache, so it works offline.

let MODELS_STATE = null;
let modelsPollTimer = null;

function formatBytes(value) {
  if (value === null || value === undefined) return "unknown";
  let size = Number(value);
  for (const unit of ["B", "KB", "MB", "GB", "TB"]) {
    if (size < 1024 || unit === "TB") return unit === "B" ? `${Math.round(size)} B` : `${size.toFixed(1)} ${unit}`;
    size /= 1024;
  }
  return `${size.toFixed(1)} TB`;
}

function formatDuration(seconds) {
  if (!seconds) return "--";
  const minutes = Math.floor(seconds / 60);
  if (minutes >= 60) return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
  return minutes ? `${minutes} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s` : `${Math.round(seconds)} s`;
}

async function loadModels() {
  try {
    MODELS_STATE = await fetchJSON("/api/models");
  } catch {
    return null; // best-effort: the footer just keeps its last label
  }
  renderModels();
  return MODELS_STATE;
}

function renderModelsFooter() {
  const button = el("models-open");
  if (!MODELS_STATE) return;
  const missing = MODELS_STATE.total_count - MODELS_STATE.ready_count;
  if (MODELS_STATE.running) {
    const total = MODELS_STATE.run_total_bytes;
    const percent = total ? Math.min(Math.round((MODELS_STATE.done_bytes / total) * 100), 99) : null;
    button.textContent = percent === null ? "Downloading models…" : `Downloading models ${percent}%`;
  } else {
    button.textContent = missing ? `Prepare models (${missing} missing)` : "Models ready";
  }
}

function modelStateText(entry) {
  if (entry.state === "downloading") return "Downloading…";
  if (entry.state === "pending") return "Queued";
  if (entry.state === "failed") return "Failed";
  return entry.cached ? "Ready" : "Not downloaded";
}

function renderModels() {
  renderModelsFooter();
  if (!MODELS_STATE || el("models-modal-overlay").hidden) return;
  const state = MODELS_STATE;
  const missing = state.models.filter((model) => !model.cached);
  const sizeText = state.sizes_known
    ? formatBytes(state.missing_bytes)
    : state.sizes_pending
      ? "sizes still loading"
      : `at least ${formatBytes(state.missing_bytes)}; the Hub did not answer for the rest`;
  el("models-summary").textContent =
    `${state.ready_count} of ${state.total_count} models ready` +
    (missing.length ? ` · ${missing.length} to download (${sizeText})` : " · nothing to download");
  el("models-download").hidden = state.running || !missing.length;
  el("models-stop").hidden = !state.running;

  const progress = el("models-progress");
  progress.hidden = !state.running;
  if (state.running) {
    const total = state.run_total_bytes;
    const percent = total ? Math.min((state.done_bytes / total) * 100, 100) : 0;
    el("models-bar-fill").style.width = `${percent}%`;
    progress.setAttribute("aria-valuenow", String(Math.round(percent)));
    const current = state.current ? (state.models.find((model) => model.key === state.current) || {}).label : null;
    el("models-progress-text").textContent =
      (current ? `${current} · ` : "") +
      `${formatBytes(state.done_bytes)} of ${formatBytes(total)} · ${formatBytes(state.bytes_per_second)}/s · ` +
      `${formatDuration(state.eta_seconds)} left` +
      (state.stopped ? " · stopping after this model" : "");
  }

  const rows = el("models-rows");
  rows.replaceChildren();
  const cell = (text, className) => {
    const td = document.createElement("td");
    td.textContent = text;
    if (className) td.className = className;
    return td;
  };
  for (const entry of state.models) {
    const row = document.createElement("tr");
    const name = cell(entry.label);
    if (entry.note) name.title = entry.note;
    const demos = entry.demos.map((id) => (demoById(id) || {}).name || id).join(", ");
    // A size somebody stated, the Hub's, or -- for the two small models a
    // library fetches by itself -- no figure at all.
    const size = cell(entry.size_bytes || entry.repo_id ? formatBytes(entry.size_bytes) : "a few MB", "models-size");
    const status = cell(modelStateText(entry), `models-state models-state-${entry.state}`);
    if (entry.error) status.title = entry.error;
    row.append(name, cell(demos, "models-demos"), size, status);
    rows.append(row);
  }
}

function startModelsPolling() {
  if (modelsPollTimer) return;
  modelsPollTimer = setInterval(async () => {
    const state = await loadModels();
    // Keep polling while a download runs, even with the dialog closed: the
    // footer button is the progress indicator then.
    if (state && !state.running && el("models-modal-overlay").hidden) stopModelsPolling();
  }, 1200);
}

function stopModelsPolling() {
  clearInterval(modelsPollTimer);
  modelsPollTimer = null;
}

async function openModelsModal() {
  el("models-modal-overlay").hidden = false;
  el("models-modal-close").focus();
  await loadModels();
  startModelsPolling();
}

function closeModelsModal() {
  el("models-modal-overlay").hidden = true;
  el("models-open").focus();
  if (!MODELS_STATE || !MODELS_STATE.running) stopModelsPolling();
}

async function startPrefetch() {
  el("models-download").disabled = true;
  try {
    await postJSON("/api/models/prefetch", {});
  } catch (err) {
    el("models-summary").textContent = err.message;
    return;
  } finally {
    el("models-download").disabled = false;
  }
  await loadModels();
  startModelsPolling();
}

async function stopPrefetch() {
  el("models-stop").disabled = true;
  try {
    await postJSON("/api/models/prefetch/stop", {});
  } finally {
    el("models-stop").disabled = false;
  }
  loadModels();
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
      label.title = "See what changed in each version";
    }
  } catch {
    // Best-effort -- an empty footer label beats breaking page load over it.
  }
}

async function init() {
  DEMOS = await fetchJSON("/api/demos");
  renderCards(DEMOS);
  loadVersion();
  showUpgradeResult().then((shown) => loadUpdateStatus({ prompt: !shown }));
  loadModels();
  initTelemetry();

  // Navigation is wired before the panels, and each panel independently:
  // one brick missing an element must not cost the whole app its routing.
  // (Wired the other way round, a single throw in wire() left every card
  // silently doing nothing when clicked, with no clue as to why.)
  window.addEventListener("hashchange", route);
  wireCompactBar();
  wireChipPanel();

  el("panel-back").addEventListener("click", () => {
    location.hash = "#/demos";
  });
  el("log-open").addEventListener("click", openLogViewer);
  // Asked before the first screen is chosen: a page opened while the loop
  // runs goes straight to its stage.
  const loopLookedAt = wireAutodemo();
  el("update-open").addEventListener("click", openUpdateFromFooter);
  el("app-version").addEventListener("click", openChangelogModal);
  el("changelog-modal-close").addEventListener("click", closeChangelogModal);
  el("changelog-modal-overlay").addEventListener("click", (event) => {
    if (event.target === el("changelog-modal-overlay")) closeChangelogModal();
  });
  el("models-open").addEventListener("click", openModelsModal);
  el("models-download").addEventListener("click", startPrefetch);
  el("models-stop").addEventListener("click", stopPrefetch);
  el("models-modal-close").addEventListener("click", closeModelsModal);
  el("models-modal-overlay").addEventListener("click", (event) => {
    if (event.target === el("models-modal-overlay")) closeModelsModal();
  });
  el("update-state").addEventListener("click", recheckForUpdates);
  el("update-now").addEventListener("click", startUpgrade);
  el("update-later").addEventListener("click", closeUpdateModal);
  el("update-modal-close").addEventListener("click", closeUpdateModal);
  el("update-modal-overlay").addEventListener("click", (event) => {
    if (event.target === el("update-modal-overlay")) closeUpdateModal();
  });
  el("log-modal-close").addEventListener("click", closeLogViewer);
  el("log-modal-overlay").addEventListener("click", (event) => {
    if (event.target === el("log-modal-overlay")) closeLogViewer();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (document.body.classList.contains("htmlc-expanded")) PANELS["html-creator"].expand(false);
    else if (!el("log-modal-overlay").hidden) closeLogViewer();
    else if (!el("update-modal-overlay").hidden) closeUpdateModal();
    else if (!el("models-modal-overlay").hidden) closeModelsModal();
    else if (!el("changelog-modal-overlay").hidden) closeChangelogModal();
    else if (!el("autodemo-modal-overlay").hidden) closeAutodemoModal();
    else if (currentPanel && !["TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) location.hash = "#/demos";
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
  await loopLookedAt;
  route();
}

init();
