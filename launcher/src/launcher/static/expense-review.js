/* Server-backed review; original extraction never becomes an approval automatically. */
const ExpenseReviewUI = (() => {
  let data = null, selected = null, dirty = false, busy = false, fetching = false;
  const root = () => document.getElementById("expx-review");
  function node(tag, text, className) {
    const n = document.createElement(tag);
    if (text) n.textContent = text;
    if (className) n.className = className;
    return n;
  }
  function button(text, action, disabled = false) {
    const b = node("button", text, "btn btn-secondary");
    b.type = "button"; b.disabled = disabled; b.onclick = action; return b;
  }
  function message(text) {
    const box = document.getElementById("expense-review-message");
    if (box) box.textContent = text;
  }
  async function request(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const problem = await res.json();
      throw new Error(problem.error || "Could not save the review");
    }
    return res;
  }
  async function refresh(force = false) {
    if (!root() || fetching || busy || (dirty && !force)) return;
    fetching = true;
    try {
      const next = await (await request("/api/expense-extract/review")).json();
      if (force || !data || data.batch_id !== next.batch_id || data.revision !== next.revision || data.phase !== next.phase) {
        if (dirty && !force) return;
        data = next; dirty = false; render();
      }
    } catch (err) { message(err.message); }
    finally { fetching = false; }
  }
  function choose(id) {
    if (dirty && !confirm("Discard unsaved field changes for this receipt?")) return;
    selected = id; dirty = false; render();
  }
  async function save(status, advance) {
    if (busy) return;
    const form = document.getElementById("expense-review-form");
    const fields = Object.fromEntries(["vendor", "date", "amount", "currency", "category"].map(key => [key, form.elements[key].value]));
    busy = true;
    form.querySelectorAll("button").forEach(b => b.disabled = true);
    try {
      data = await (await request(`/api/expense-extract/review/${selected}`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({batch_id: data.batch_id, revision: data.revision, fields, status,
          duplicate_note: form.elements.duplicate_note.value})
      })).json();
      dirty = false;
      if (advance) selected = data.items.find(i => i.status === "pending" && i.id !== selected)?.id || selected;
      render(); message(status === "approved" ? "Approved and saved locally." : "Review saved locally.");
      if (advance) document.getElementById("expense-review-form").elements.vendor.focus();
    } catch (err) {
      message(err.message);
      document.getElementById("expense-review-message").scrollIntoView({block: "center"});
    }
    finally { busy = false; form.querySelectorAll("button").forEach(b => b.disabled = false); }
  }
  async function download() {
    if (dirty) { message("Save your field changes before exporting."); return; }
    try {
      const response = await request(`/api/expense-extract/review/${data.batch_id}/export?revision=${data.revision}`);
      const url = URL.createObjectURL(await response.blob());
      const a = node("a"); a.href = url; a.download = "reviewed-expenses.csv"; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      message("Exported approved receipts only. Pending and excluded receipts were omitted.");
    } catch (err) { message(err.message); }
  }
  function render() {
    const host = root(); host.replaceChildren();
    host.append(node("h2", "Review expenses"));
    const notice = node("p", "Compare each image with its fields, then approve or exclude it. Reviews are saved on this computer and survive a restart. Starting a new batch replaces this review.", "expense-help");
    host.append(notice);
    const feedback = node("p"); feedback.id = "expense-review-message"; feedback.setAttribute("role", "status");
    host.append(feedback);
    if (!data?.items.length) { host.append(node("p", "Choose a receipt folder above and press Start to begin.")); return; }
    const counts = {approved: 0, pending: 0, excluded: 0};
    data.items.forEach(i => counts[i.status]++);
    const bar = node("div", null, "expense-review-bar");
    bar.append(node("strong", `${counts.approved} approved · ${counts.pending} pending · ${counts.excluded} excluded`));
    bar.append(button("Export approved CSV", download, !counts.approved || data.phase === "extracting"));
    bar.append(button("Reload saved review", () => {
      if (!dirty || confirm("Discard unsaved edits and reload the saved review?")) refresh(true);
    }));
    host.append(bar);
    host.append(node("p", "Approved totals: " + (Object.entries(data.totals).map(([c, a]) => `${c} ${a}`).join(" · ") || "none yet"), "expense-totals"));
    if (data.phase !== "complete") host.append(node("p", data.phase === "extracting"
      ? "Extracting… You can inspect receipts now. Editing becomes available when extraction finishes or stops."
      : `Extraction ${data.phase}. This may be a partial batch; receipts without extracted text can be entered manually.`, "expense-warning"));
    const item = data.items.find(i => i.id === selected) || data.items.find(i => i.status === "pending") || data.items[0];
    selected = item.id;
    const layout = node("div", null, "expense-review-layout");
    const list = node("nav", null, "expense-receipt-list"); list.setAttribute("aria-label", "Receipts in this batch");
    data.items.forEach(i => {
      const b = button(`${i.source_file}\n${i.status}${i.duplicates.length ? " · Possible duplicate" : ""}`, () => choose(i.id));
      b.classList.toggle("selected", i.id === selected); b.setAttribute("aria-current", i.id === selected ? "true" : "false"); list.append(b);
    });
    const imagePanel = node("div", null, "expense-image-panel");
    const link = node("a", "Open full-size receipt"); link.href = item.image_url; link.target = "_blank"; link.rel = "noopener";
    const image = node("img"); image.src = item.image_url; image.alt = `Original receipt: ${item.source_file}`;
    image.onerror = () => { image.hidden = true; imagePanel.append(node("p", "Receipt unavailable or changed. Check the source file before approving.", "expense-warning")); };
    imagePanel.append(link, image);
    const editor = node("div"); editor.append(node("h3", item.source_file));
    if (item.duplicates.length) editor.append(node("p", `Possible duplicate of: ${item.duplicates.join(", ")}. Exclude this receipt or explain why it is a separate expense.`, "expense-warning"));
    const original = item.original;
    if (!original) editor.append(node("p", "No extraction result yet. Enter the fields from the image when extraction stops.", "expense-warning"));
    else if (original.error || original.review_reasons.length) editor.append(node("p", "Original extraction flags: " + [original.error, ...original.review_reasons].filter(Boolean).join(" · "), "expense-warning"));
    const form = node("form"); form.id = "expense-review-form"; form.onsubmit = e => e.preventDefault();
    const fieldset = node("fieldset"); fieldset.disabled = data.phase === "extracting";
    for (const [key, label] of [["vendor", "Vendor"], ["date", "Date"], ["amount", "Amount"], ["currency", "Currency"], ["category", "Category"]]) {
      const wrapper = node("label", null, "field"); wrapper.append(node("span", label));
      const choices = key === "currency" ? data.currencies : key === "category" ? data.categories : null;
      const input = node(choices ? "select" : "input"); input.name = key;
      if (choices) ["", ...choices].forEach(value => { const option = node("option", value || "Choose…"); option.value = value; input.append(option); });
      else { input.type = key === "date" ? "date" : "text"; input.maxLength = 250; }
      if (key === "amount") { input.inputMode = "decimal"; input.placeholder = "426.00"; }
      input.value = item.fields[key] || ""; wrapper.append(input); fieldset.append(wrapper);
    }
    fieldset.append(node("small", "Amounts use a decimal point, without thousands separators. Refunds are negative."));
    const noteLabel = node("label", null, "field"); noteLabel.append(node("span", "Duplicate decision / review note"));
    const note = node("textarea"); note.name = "duplicate_note"; note.maxLength = 1000; note.value = item.duplicate_note; noteLabel.append(note); fieldset.append(noteLabel);
    const actions = node("div", null, "expense-review-actions");
    actions.append(button("Save pending", () => save("pending", false)), button("Exclude", () => save("excluded", true)));
    const approve = button("Approve & next", () => save("approved", true)); approve.className = "btn btn-primary";
    actions.append(approve); fieldset.append(actions); form.append(fieldset);
    form.oninput = () => { dirty = true; message("Unsaved changes. Save pending or approve to keep them."); };
    editor.append(form);
    const source = node("details"); source.append(node("summary", "Original extraction & OCR text"));
    source.append(node("pre", original ? JSON.stringify(Object.fromEntries(["vendor", "date", "amount", "currency", "category"].map(k => [k, original[k]])), null, 2) + "\n\n" + original.raw_text : "No OCR result available."));
    editor.append(source); layout.append(list, imagePanel, editor); host.append(layout);
  }
  setInterval(() => {
    if (root() && !root().closest("[hidden]") && !dirty) refresh();
  }, 2500);
  return {refresh, confirmNewBatch: () => !data?.items.length || confirm("Starting a new batch replaces the saved review. Export approved receipts first if you need them. Continue?")};
})();
