// Durable report review. Extraction messages only trigger a refresh; the server
// snapshot is authoritative, so reconnecting cannot duplicate receipts or lose edits.
class ExpenseReview {
  // `onCleared` runs once the results have left the view, so the panel can
  // reset what it owns (its progress lines, its status).
  constructor({ onCleared = () => {} } = {}) {
    this.onCleared = onCleared;
    this.data = null;
    this.selected = null;
    this.dirty = false;
    this.loading = 0;
    this.saving = false;
    this.reportId = null;
    el("expx-edit-form").addEventListener("input", () => {
      this.dirty = true;
      el("expx-discard").disabled = false;
      el("expx-export").disabled = true;
      el("expx-edit-status").textContent = "Unsaved changes. Save a draft or validate this expense.";
    });
    el("expx-edit-form").addEventListener("submit", (event) => { event.preventDefault(); this.save(true); });
    el("expx-save").addEventListener("click", () => this.save(false));
    el("expx-discard").addEventListener("click", async () => {
      if (!this.selected || this.saving) return;
      try {
        const data = await fetchJSON(`/api/expense-extract/report?report_id=${this.selected.report_id}`);
        const item = data.items.find((item) => item.id === this.selected.id);
        this.data = data;
        this.open(item);
        this.render();
      } catch (error) {
        el("expx-edit-status").textContent = error.message;
      }
    });
    el("expx-report").addEventListener("change", () => {
      const id = el("expx-report").value;
      if (!id) return; // the "open a saved report" line, shown while nothing is open
      if (!this.canLeave()) {
        el("expx-report").value = this.data?.report?.id || "";
        return;
      }
      this.selected = null;
      this.refresh(id);
    });
    el("expx-export").addEventListener("click", () => this.export());
    el("expx-clear").addEventListener("click", () => this.clear());
    el("expx-delete").addEventListener("click", () => this.deleteAll());
    el("expx-receipt").addEventListener("error", () => {
      el("expx-receipt").hidden = true;
      el("expx-image-error").hidden = false;
    });
    window.addEventListener("beforeunload", (event) => {
      if (this.dirty) { event.preventDefault(); event.returnValue = ""; }
    });
  }

  canLeave() {
    if (!this.dirty && !this.saving) return true;
    el("expx-edit-status").textContent = "Save this expense before switching receipts, reports, or starting a new batch.";
    return false;
  }

  async refresh(reportId = this.reportId) {
    this.reportId = reportId;
    const request = ++this.loading;
    try {
      const data = await fetchJSON(`/api/expense-extract/report${reportId ? `?report_id=${encodeURIComponent(reportId)}` : ""}`);
      if (request !== this.loading) return;
      this.data = data;
      this.reportId = data.report?.id;
      this.render();
    } catch (error) {
      if (request !== this.loading) return;
      // A report deleted from another tab: fall back to whatever is current.
      if (reportId) return this.refresh(null);
      el("expx-summary").textContent = `Could not load report: ${error.message}`;
    }
  }

  // Empty the view for the next run. Nothing is deleted: the report stays
  // under Saved report.
  async clear() {
    if (!this.canLeave()) return;
    try {
      await fetchJSON("/api/expense-extract/report/close", { method: "POST" });
    } catch (error) {
      el("expx-summary").textContent = error.message;
      return;
    }
    this.selected = null;
    await this.refresh(null);
    this.onCleared();
  }

  // Delete every saved report, after asking: validated work goes with them.
  async deleteAll() {
    const count = this.data?.reports.length || 0;
    if (!count || !this.canLeave()) return;
    const what = count === 1 ? "the saved expense report" : `all ${count} saved expense reports`;
    if (!confirm(`Delete ${what}, including every validated expense? Your receipt images are not touched. This can't be undone.`)) return;
    try {
      await fetchJSON("/api/expense-extract/reports", { method: "DELETE" });
    } catch (error) {
      el("expx-summary").textContent = error.message;
      return;
    }
    this.selected = null;
    await this.refresh(null);
    this.onCleared();
  }

  render() {
    const data = this.data;
    const open = Boolean(data.report);
    const validated = data.items.filter((item) => item.status === "validated").length;
    const totals = Object.entries(data.totals).map(([currency, amount]) => `${currency} ${amount}`).join(" · ");
    if (open) el("expx-summary").textContent = `${validated}/${data.items.length} validated${totals ? ` · ${totals}` : ""}`;
    else el("expx-summary").textContent = data.reports.length ? "No report open." : "Read receipts to build a report.";
    el("expx-export").disabled = this.exportDisabled();
    el("expx-clear").disabled = !open || data.running;
    el("expx-delete").hidden = !data.reports.length;
    el("expx-delete").disabled = data.running;
    const saved = data.reports.map((report) => ({value: report.id,
      label: `${new Date(report.created).toLocaleString()} · ${report.folder.split(/[\\/]/).pop()}`}));
    // With nothing open the first line says so; picking a report opens it.
    fillSelect(el("expx-report"), open ? saved : [{value: "", label: saved.length ? "Open a saved report…" : "No reports yet"}, ...saved]);
    el("expx-report").value = open ? data.report.id : "";
    const list = el("expx-list");
    list.replaceChildren();
    for (const item of data.items) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "expense-card";
      button.setAttribute("aria-pressed", String(this.selected?.id === item.id));
      const title = document.createElement("strong");
      title.textContent = item.vendor || item.source_file;
      const amount = document.createElement("span");
      amount.textContent = `${item.currency || "Currency?"} ${item.amount ?? "Amount?"} · ${item.date || "Date?"}`;
      const status = document.createElement("span");
      status.className = `expense-badge ${item.status}`;
      status.textContent = item.status === "validated" ? "Validated" : "To review";
      button.append(title, amount, status);
      button.addEventListener("click", () => {
        if (this.selected?.id === item.id || !this.canLeave()) return;
        this.open(item);
        this.render();
      });
      list.append(button);
    }
    if (!data.items.length) {
      showPlaceholder(list, "Receipts will appear here as they are read.");
      el("expx-detail").hidden = true;
      this.selected = null;
    } else if (!this.selected || !data.items.some((item) => item.id === this.selected.id)) {
      this.open(data.items[0]);
      list.firstElementChild?.setAttribute("aria-pressed", "true");
    } else if (!this.dirty && !this.saving) {
      const latest = data.items.find((item) => item.id === this.selected.id);
      if (latest.revision !== this.selected.revision) this.open(latest);
    }
  }

  exportDisabled() {
    return !this.data?.items.length || this.dirty || this.saving ||
      (this.data.running && this.data.active_report_id === this.data.report?.id);
  }

  open(item) {
    this.selected = item;
    this.dirty = false;
    el("expx-discard").disabled = true;
    el("expx-detail").hidden = false;
    el("expx-detail-title").textContent = item.source_file;
    fillSelect(el("expx-edit-currency"), [{value: "", label: "Choose currency"},
      ...this.data.currencies.map((currency) => ({value: currency, label: currency}))]);
    fillSelect(el("expx-edit-category"), this.data.categories.map((category) => ({value: category, label: category})));
    for (const key of ["vendor", "date", "amount", "currency", "category", "notes"]) {
      el(`expx-edit-${key}`).value = item[key] ?? "";
    }
    el("expx-raw-text").textContent = item.original.raw_text || "No text extracted. Complete the details manually from the receipt.";
    const warnings = [item.original.error, ...(item.original.review_reasons || [])].filter(Boolean);
    el("expx-review-warning").textContent = warnings.length ? `Original extraction notes: ${warnings.join("; ")}.` : "Check the details against the receipt before validating.";
    el("expx-edit-status").textContent = item.status === "validated" ? "Validated. Saving changes as a draft will reopen review." : "Awaiting your review.";
    const url = `/api/expense-extract/reports/${item.report_id}/expenses/${item.id}/receipt`;
    el("expx-receipt").hidden = false;
    el("expx-image-error").hidden = true;
    el("expx-receipt").src = url;
    el("expx-receipt-link").href = url;
  }

  async save(validate) {
    if (!this.selected || this.saving) return;
    this.saving = true;
    const fields = [...el("expx-edit-form").querySelectorAll("input, select, textarea, button")];
    fields.forEach((field) => { field.disabled = true; });
    el("expx-export").disabled = true;
    const item = this.selected;
    const body = {revision: item.revision, validate};
    for (const key of ["vendor", "date", "amount", "currency", "category", "notes"]) body[key] = el(`expx-edit-${key}`).value;
    try {
      const saved = await fetchJSON(`/api/expense-extract/reports/${item.report_id}/expenses/${item.id}`, {
        method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
      });
      this.open(saved);
      await this.refresh(item.report_id);
      el("expx-edit-status").textContent = validate ? "Expense validated and saved." : "Draft saved.";
    } catch (error) {
      el("expx-edit-status").textContent = error.message;
    } finally {
      this.saving = false;
      fields.forEach((field) => { field.disabled = false; });
      el("expx-discard").disabled = !this.dirty;
      el("expx-export").disabled = this.exportDisabled();
    }
  }

  async export() {
    if (!this.canLeave() || !this.data?.report) return;
    el("expx-export").disabled = true;
    try {
      const response = await fetch(`/api/expense-extract/reports/${this.data.report.id}/export.xlsx`);
      if (!response.ok) throw new Error((await response.json()).error || "Export failed");
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "expense-report.xlsx";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (error) {
      el("expx-summary").textContent = error.message;
    } finally {
      el("expx-export").disabled = this.exportDisabled();
    }
  }
}
