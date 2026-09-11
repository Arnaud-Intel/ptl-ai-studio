"""Local receipt review state, independent of model output and socket delivery."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import threading
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path

from expense_extract.parsing import CURRENCIES
from .errors import Conflict

CATEGORIES = ("Meals", "Travel", "Lodging", "Office Supplies", "Software", "Other")
FIELDS = ("vendor", "date", "amount", "currency", "category")


class ExpenseReview:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.data = {"batch_id": "", "revision": 0, "phase": "empty", "items": []}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data["phase"] == "extracting":
                self.data["phase"] = "interrupted"

    def _save(self):
        self.data["revision"] += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.path)

    def begin(self, images):
        with self.lock:
            if self.data["phase"] == "extracting":
                raise Conflict("Extraction is already running")
            items = []
            for image in images:
                resolved = image.resolve()
                if resolved.parent != image.parent.resolve() or not resolved.is_file():
                    raise ValueError("Receipt images must be files inside the selected folder")
                items.append({"id": uuid.uuid4().hex, "source_file": image.name,
                              "path": str(resolved), "hash": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                              "original": None, "fields": dict.fromkeys(FIELDS, ""),
                              "status": "pending", "duplicate_note": ""})
            self.data = {"batch_id": uuid.uuid4().hex, "revision": 0,
                         "phase": "extracting", "items": items}
            self._save()

    def add(self, line):
        with self.lock:
            item = next(i for i in self.data["items"] if i["source_file"] == line.source_file)
            item["original"] = line.to_dict()
            item["fields"] = {key: line.to_dict().get(key) or "" for key in FIELDS}
            self._save()

    def finish(self, phase):
        with self.lock:
            if phase == "complete" and any(i["original"] is None for i in self.data["items"]):
                phase = "partial"
            self.data["phase"] = phase
            self._save()

    def _duplicates(self, item):
        fields = item["fields"]
        def signature(f):
            try:
                amount = Decimal(f["amount"])
                if not amount.is_finite() or not f["vendor"].strip() or not f["date"] or not f["currency"]:
                    return None
                return (" ".join(f["vendor"].casefold().split()), f["date"], amount, f["currency"])
            except Exception:
                return None
        key = signature(fields)
        return [other["source_file"] for other in self.data["items"]
                if other["id"] != item["id"] and other["status"] != "excluded"
                and (other["hash"] == item["hash"] or (key is not None and key == signature(other["fields"])))]

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.data)
            totals = {}
            for item, source in zip(result["items"], self.data["items"]):
                item.pop("path")
                item.pop("hash")
                item["duplicates"] = self._duplicates(source)
                item["image_url"] = f'/api/expense-extract/review/{result["batch_id"]}/{item["id"]}/image'
                if item["status"] == "approved":
                    f = item["fields"]
                    totals[f["currency"]] = totals.get(f["currency"], Decimal(0)) + Decimal(f["amount"])
            result["totals"] = {key: format(value, "f") for key, value in sorted(totals.items())}
            result["currencies"] = sorted(CURRENCIES)
            result["categories"] = CATEGORIES
            return result

    def update(self, batch_id, item_id, revision, fields, status, duplicate_note):
        with self.lock:
            self._check(batch_id)
            if revision != self.data["revision"]:
                raise Conflict("The review changed in another tab. Reload the review before saving.")
            if self.data["phase"] == "extracting":
                raise Conflict("Wait for extraction to finish, or stop it before reviewing.")
            item = self._item(item_id)
            if status not in {"pending", "approved", "excluded"}:
                raise ValueError("Invalid review status")
            fields = {key: str(fields.get(key) or "").strip() for key in FIELDS}
            fields["currency"] = fields["currency"].upper()
            if any(len(value) > 250 for value in fields.values()) or len(duplicate_note) > 1000:
                raise ValueError("Review fields are too long")
            if status == "approved":
                self.image(batch_id, item_id)
                if not fields["vendor"] or fields["category"] not in CATEGORIES:
                    raise ValueError("Enter a vendor and choose a category")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["date"]):
                    raise ValueError("Enter a valid date in YYYY-MM-DD format")
                date.fromisoformat(fields["date"])
                if fields["currency"] not in CURRENCIES:
                    raise ValueError("Choose a supported currency")
                places = 0 if fields["currency"] in {"JPY", "KRW"} else 2
                pattern = r"-?\d{1,12}" + (r"(?:\.\d{1,2})?" if places else "")
                if not re.fullmatch(pattern, fields["amount"]):
                    raise ValueError("Enter an amount with a decimal point, no grouping, and valid currency precision")
                fields["amount"] = format(Decimal(fields["amount"]), f".{places}f")
            candidate = {**item, "fields": fields}
            if status == "approved" and self._duplicates(candidate) and not duplicate_note.strip():
                raise ValueError("Possible duplicate: exclude it or explain why this is a separate expense")
            # A changed match must be reviewed again, even if an older row was approved.
            before = {i["id"]: self._duplicates(i) for i in self.data["items"]}
            item.update(fields=fields, status=status, duplicate_note=duplicate_note.strip())
            for other in self.data["items"]:
                if other is not item and other["status"] == "approved":
                    if set(self._duplicates(other)) - set(before[other["id"]]):
                        other["status"] = "pending"
                        other["duplicate_note"] = ""
            self._save()
            return self.snapshot()

    def _check(self, batch_id):
        if batch_id != self.data["batch_id"]:
            raise Conflict("This batch has been replaced. Reload the review.")

    def _item(self, item_id):
        item = next((i for i in self.data["items"] if i["id"] == item_id), None)
        if item is None:
            raise ValueError("Unknown receipt")
        return item

    def image(self, batch_id, item_id):
        with self.lock:
            self._check(batch_id)
            item = self._item(item_id)
            path = Path(item["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
                raise Conflict("The source image changed. Extract the folder again before reviewing.")
            return path

    def export(self, batch_id, revision):
        with self.lock:
            self._check(batch_id)
            if revision != self.data["revision"] or self.data["phase"] == "extracting":
                raise Conflict("Reload the completed review before exporting")
            approved = [i for i in self.data["items"] if i["status"] == "approved"]
            if not approved:
                raise ValueError("Approve at least one receipt before exporting")
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow(["source_file", *FIELDS, "review_status", "duplicate_note"])
            def safe(value):
                return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value
            for item in approved:
                self.image(batch_id, item["id"])
                writer.writerow([safe(item["source_file"]), *[
                    item["fields"][key] if key == "amount" else safe(item["fields"][key]) for key in FIELDS
                ], "approved", safe(item["duplicate_note"])])
            return "\ufeff" + stream.getvalue()
