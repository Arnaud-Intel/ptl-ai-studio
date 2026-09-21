# Scanned and photographed receipts

Five fictional receipts generated as individual photographic/scan images with
the built-in image-generation tool. These are demo assets, not evidence of real
purchases. Each image includes a small FICTIONAL DEMO notice.

Select **Scanned & photographed · 5 worn receipts** in Expense Report Extractor.
The original clean receipt packs remain available for comparison.

| Receipt | What makes it harder | Expected total (EUR) |
| --- | --- | ---: |
| La Lanterne Cafe | Skew, thermal-print dropout, fold and dark tabletop | 38.40 |
| Rive Taxi | Scanner streak, diagonal crease, faded print | 25.60 |
| Atelier Quay Hotel | Perspective, folds, shadow and coffee stain; paid amount vs zero balance | 406.00 |
| Workbench Papeterie purchase | Crumpled paper on fabric, faded heading; subtract the discount | 32.30 |
| Workbench Papeterie refund | Fold, worn paper and refund stamp; retain the negative sign | -18.00 |
| **Net total** | | **484.30** |

Dates use French day/month/year notation. Monetary values use decimal commas.
The answer key records visually checked printed figures, not model predictions.
Image-generation artifacts and recognition errors are possible: compare the
source image during review. This pack is a demonstration, not an OCR benchmark.

For the demo, open an expense, compare its total with the receipt, correct it if
needed, validate it, and export the batch to Excel. The hotel should contribute
EUR 406.00, not its EUR 0.00 balance. The purchase should use EUR 32.30 after the
discount, not its EUR 35.30 subtotal. The refund reduces spending by EUR 18.00.

The exact generation prompts are in [scanned-receipts-prompts.md](scanned-receipts-prompts.md).
