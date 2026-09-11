"""Rebuild fictional demo documents and deterministic receipt images (Pillow).

No downloaded assets, real logos, payment credentials or private documents.
Run from any directory with Python and Pillow; outputs stay under sample-data.
"""
from pathlib import Path
from decimal import Decimal
import json
import shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / "sample-data"
EXPENSES = ROOT / "expenses"
DISCLAIMER = "FICTIONAL DEMO — NOT VALID FOR PAYMENT"


def write(relative, text):
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


RECEIPTS = [
    dict(file="01-lanterne-cafe.png", vendor="Lanterne Cafe", kind="CAFE RECEIPT", date="2026-09-03", currency="EUR", amount="38.40", category="Meals", items=[("Working lunch - 2 guests", "29.00"), ("Coffee - 2 cups", "6.40"), ("Sparkling water", "3.00")], note="Project M-FR-26 / customer workshop lunch", style="till"),
    dict(file="02-atelier-quay-hotel.png", vendor="Atelier Quay Hotel", kind="GUEST FOLIO", date="2026-09-05", currency="EUR", amount="426.00", category="Lodging", items=[("Room / 2 nights at EUR 180.00", "360.00"), ("Breakfast / 2 mornings", "36.00"), ("Local accommodation levy", "30.00")], note="Alex Morgan / arrival Sep 03 / departure Sep 05", style="invoice"),
    dict(file="03-northwind-air.png", vendor="Northwind Air Demo", kind="PASSENGER RECEIPT", date="2026-09-02", currency="EUR", amount="248.60", category="Travel", items=[("Demo route Paris - Lyon / flexible fare", "205.00"), ("Airport and service charges", "43.60")], note="Traveler: Alex Morgan / reference DEMO-AIR-026", style="ticket"),
    dict(file="04-veloria-rail.png", vendor="Veloria Rail", kind="RAIL TRAVEL RECEIPT", date="2026-09-04", currency="EUR", amount="82.00", category="Travel", items=[("Lyon - Valence / 2 demo tickets", "82.00")], note="Customer site transfer / reference DEMO-RAIL-041", style="ticket"),
    dict(file="05-rive-transfer.png", vendor="Rive Transfer", kind="TRANSFER RECEIPT", date="2026-09-03", currency="EUR", amount="46.80", category="Travel", items=[("Demo station - customer site", "38.00"), ("Evening service supplement", "8.80")], note="Pickup 19:15 / travel time 28 min / distance 14 km", style="till"),
    dict(file="06-workbench-supply.png", vendor="Workbench Supply", kind="ITEMIZED RECEIPT", date="2026-09-04", currency="EUR", amount="129.90", category="Office Supplies", items=[("USB-C demo adapter kit", "59.90"), ("Workshop headset", "70.00")], note="M-FR-26 / replacement equipment for site training", style="invoice"),
    dict(file="07-netscope-sandbox.png", vendor="NetScope Sandbox", kind="SUBSCRIPTION INVOICE", date="2026-09-01", currency="USD", amount="49.00", category="Software", items=[("Team demo workspace / September", "49.00")], note="Billing account DEMO-MERIDIAN / no real subscription", style="invoice"),
    dict(file="08-workbench-credit.png", vendor="Workbench Supply", kind="CREDIT NOTE / REFUND", date="2026-09-05", currency="EUR", amount="-18.00", category="Office Supplies", items=[("Returned spare cable / original sale DEMO-080", "-18.00")], note="Refund issued / reduces reimbursable expenses", style="till"),
    dict(file="09-maison-mistral.png", vendor="Maison Mistral", kind="DINNER RECEIPT", date="2026-09-04", currency="EUR", amount="94.50", category="Meals", items=[("Seasonal plates / 2 guests", "48.00"), ("Vegetarian main / 2 guests", "36.00"), ("Coffee and water", "10.50")], note="Alex Morgan + 3 fictional customer attendees", style="till"),
]
REVIEW = [
    dict(file="01-unknown-dollar.png", vendor="Harbor Kiosk", kind="RECEIPT / REVIEW CASE", date="2026-09-04", currency=None, amount="24.00", display="$24.00", category="Meals", items=[("Workshop refreshments", "24.00")], note="Only a dollar sign is printed. Country not specified.", style="till", reason="Currency is ambiguous; do not assume USD"),
    dict(file="02-ambiguous-separator.png", vendor="Demo Equipment Desk", kind="RECEIPT / REVIEW CASE", date="2026-09-04", currency="EUR", amount=None, display="EUR 1,234", category="Office Supplies", items=[("Equipment rental - source amount ambiguous", "1,234")], note="Separator intentionally ambiguous; do not normalize blindly.", style="invoice", reason="Amount 1,234 is ambiguous"),
    dict(file="03-invalid-date.png", vendor="Lanterne Cafe", kind="RECEIPT / REVIEW CASE", date="2026-02-30", currency="EUR", amount="18.50", category="Meals", items=[("Breakfast meeting", "18.50")], note="The printed calendar date is intentionally invalid.", style="till", reason="Printed date does not exist"),
]


def font(size, bold=False):
    candidates = [Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def receipt(entry, folder, index):
    # Exact text and sums are part of the fixture, unlike generative artwork.
    image = Image.new("RGB", (1100, 1450), "#e8e8e3")
    draw = ImageDraw.Draw(image)
    till = entry["style"] == "till"
    left, right = (170, 930) if till else (60, 1040)
    draw.rectangle((left + 9, 49, right + 9, 1409), fill="#c9ccc8")
    draw.rectangle((left, 40, right, 1400), fill="#fffef9")
    accent = {"till": "#313733", "invoice": "#163e50", "ticket": "#87452e"}[entry["style"]]
    draw.rectangle((left, 40, right, 98), fill=accent)
    draw.text((left + 24, 59), "FICTIONAL DEMO / NOT FOR PAYMENT", font=font(23, True), fill="white")
    x = left + 38
    draw.text((x, 140), entry["vendor"], font=font(39, True), fill=accent)
    draw.text((x, 204), entry["kind"], font=font(23, True), fill=accent)
    draw.text((x, 258), "14 Atelier Lane / Demo District", font=font(23), fill="#4a5556")
    draw.text((x, 296), "Fictional address / all parties invented", font=font(21), fill="#4a5556")
    draw.line((x, 355, right - 38, 355), fill=accent, width=2)
    draw.text((x, 390), f"Date: {entry['date']}", font=font(27, True), fill="#202a2e")
    draw.text((x, 437), f"Document: DEMO-{index:03d} / PAID (SIMULATED)", font=font(22), fill="#34434b")
    draw.text((x, 480), "Billed to: Meridian Robotics / Alex Morgan", font=font(22), fill="#34434b")
    y = 563
    for label, amount in entry["items"]:
        # Wrap at words with measured width, leaving amount on its own right-aligned row.
        words, line = label.split(), ""
        for word in words:
            candidate = (line + " " + word).strip()
            if draw.textlength(candidate, font=font(25)) > right - x - 70:
                draw.text((x, y), line, font=font(25), fill="#222d30")
                y += 34
                line = word
            else:
                line = candidate
        draw.text((x, y), line, font=font(25), fill="#222d30")
        y += 38
        currency = entry["currency"] or "$"
        displayed = amount.replace(".", ",") if currency == "EUR" else amount
        draw.text((right - 38, y), f"{currency} {displayed}", font=font(27, True), anchor="ra", fill="#222d30")
        y += 63
    y = max(y + 18, 963)
    draw.rectangle((x - 12, y, right - 26, y + 119), fill="#eef1ed")
    draw.text((x, y + 15), "TOTAL REFUND" if (entry.get("amount") or "").startswith("-") else "TOTAL PAID", font=font(25, True), fill=accent)
    amount = entry.get("display") or f"{entry['currency']} {entry['amount'].replace('.', ',') if entry['currency'] == 'EUR' else entry['amount']}"
    draw.text((right - 38, y + 60), amount, font=font(39, True), anchor="ra", fill=accent)
    draw.text((x, 1160), "Taxes included where applicable / no payment due", font=font(20), fill="#465153")
    note = entry["note"]
    # Two short lines, never tiny print or text clipped outside a receipt.
    words, lines, line = note.split(), [], ""
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=font(20)) > right - x - 40:
            lines.append(line); line = word
        else:
            line = candidate
    lines.append(line)
    for offset, text in enumerate(lines):
        draw.text((x, 1210 + offset * 29), text, font=font(20), fill="#465153")
    draw.line((x, 1322, right - 38, 1322), fill="#bec6c1", width=2)
    draw.text((x, 1350), "SYNTHETIC SAMPLE / NO REAL TRANSACTION", font=font(20, True), fill=accent)
    folder.mkdir(parents=True, exist_ok=True)
    image.save(folder / entry["file"], optimize=True)


def main():
    for i, entry in enumerate(RECEIPTS, 1):
        assert sum(Decimal(amount) for _, amount in entry["items"]) == Decimal(entry["amount"])
        receipt(entry, EXPENSES / "customer-rollout", i)
    for i, entry in enumerate(REVIEW, 101):
        receipt(entry, EXPENSES / "needs-review", i)
    quick = EXPENSES / "quick-start"
    quick.mkdir(parents=True, exist_ok=True)
    for name in ["01-lanterne-cafe.png", "05-rive-transfer.png", "06-workbench-supply.png"]:
        shutil.copyfile(EXPENSES / "customer-rollout" / name, quick / name)
    truth = {"fictional": True, "currency_totals": {"EUR": "1048.20", "USD": "49.00"},
             "quick_start_totals": {"EUR": "215.10"}, "receipts": RECEIPTS, "review_cases": REVIEW,
             "note": "Ground truth, not measured OCR/model results. Keep outside folders ingested as receipts."}
    write("expenses/expected-results.json", json.dumps(truth, indent=2, ensure_ascii=False))
    write("expenses/README.md", '''# Meridian customer rollout — fictional expense kit

All vendors, people, addresses, identifiers and transactions are invented. Every
image is marked as a synthetic demo and is invalid for payment or reimbursement.

Alex Morgan visits a fictional customer workshop in Lyon, September 3–5, 2026,
under project M-FR-26. The pack includes a hotel folio, flight and rail tickets,
transfers, meals, equipment, a software subscription and a refund.

| Pack | Content | Expected totals |
| --- | --- | --- |
| Quick start | 3 clear receipts: cafe, transfer, office supplies | EUR 215.10 |
| Customer rollout | 9 receipts; 8 EUR and 1 USD; includes EUR -18.00 refund | EUR 1,048.20 and USD 49.00, separately |
| Needs review | 3 deliberate exceptions: ambiguous $, ambiguous 1,234, invalid date | All need review; do not aggregate |

Open Expense Report Extractor → choose a pack → Start. Inspect the preview cards
before running. The answer key is an independent fixture, not generated results.
Models may misread or omit values; compare their output with expected-results.json.
No exchange rate is supplied, so a combined converted total would be invented.
''')
    thumbs = Image.new("RGB", (1200, 1320), "#ecefe9")
    for i, entry in enumerate(RECEIPTS + REVIEW):
        source = EXPENSES / ("customer-rollout" if i < 9 else "needs-review") / entry["file"]
        thumb = Image.open(source)
        thumb.thumbnail((280, 400))
        thumbs.paste(thumb, ((i % 4) * 300 + 10, (i // 4) * 440 + 10))
    thumbs.save(EXPENSES / "contact-sheet.jpg", quality=90)


if __name__ == "__main__":
    main()
