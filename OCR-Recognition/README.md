````markdown
# Project 4 — Image/Text Recognition (Basic)
**DecodeLabs AI Industrial Training, Batch 2026 — Optional Mastery Phase**

This project fully implements the **Path 1: OCR** requirements provided in the PDF using `pytesseract` (Tesseract OCR engine). The complete pipeline has been tested successfully, and the test results are provided below.

## Files
```text
ocr_project/
├── generate_sample.py     # Synthetic "messy" invoice image generator (noise + shadow + skew)
├── ocr_pipeline.py        # MAIN script — Project 4 deliverable
├── sample_input/
│   ├── messy_invoice.png  # Sample 1: noisy/skewed invoice
│   └── book_page.png      # Sample 2: paragraph text (robustness test)
└── outputs/               # Generated results after each run
    ├── *_1_binary.png     # Pre-processed (deskewed + adaptive thresholded) image
    ├── *_2_confirmed.png  # Visual confirmation: green box = accepted, red = rejected
    ├── *_extracted.txt    # Final extracted text
    └── *_report.json      # Full machine-readable report
````

---

## Local Machine Setup

### Step 1 — Install Tesseract OCR Engine

Tesseract is a **system-level dependency**, separate from Python.

* **Windows:** Download and install the UB-Mannheim Tesseract installer from:
  `https://github.com/UB-Mannheim/tesseract/wiki`

  The default installation path is:
  `C:\Program Files\Tesseract-OCR`

  If Tesseract is not automatically added to PATH, add the following line at the top of `ocr_pipeline.py`:

  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

* **Mac:**

  ```bash
  brew install tesseract
  ```

* **Linux:**

  ```bash
  sudo apt install tesseract-ocr
  ```

### Step 2 — Install Python Dependencies

Open a terminal/CMD inside the project folder and run:

```bash
pip install opencv-python pytesseract numpy pillow
```

### Step 3 — Run the Sanity Check

Run the bundled sample:

```bash
python ocr_pipeline.py
```

If everything is configured correctly, the terminal will display the OCR report and the final line will show:

```text
[GATE] PASSED
```

If you receive an error such as `tesseract is not installed or it's not in your PATH`, verify the Tesseract installation from Step 1.

### Step 4 — Run OCR on Your Own Image

```bash
python ocr_pipeline.py path/to/your_image.png --psm sparse
```

Available `--psm` modes:

* `auto` — Mixed or complex layouts
* `block` — Paragraph/block text
* `line` — Single-line text such as number plates or headers
* `sparse` — Invoices and scattered text

### Step 5 — Verify the Output

Each run generates four output files inside the `outputs/` folder.

The most important file for demonstration/defense is:

```text
*_2_confirmed.png
```

It provides a visual confirmation of the confidence gate:

* **Green box** → Accepted text (confidence ≥ 80%)
* **Red box** → Rejected text (confidence < 80%)

---

## Test Results

| Test | Input                                            | PSM    | Tokens Detected | Tokens **Accepted (≥80%)** | Avg. Confidence | Gate                                |
| ---- | ------------------------------------------------ | ------ | --------------: | -------------------------: | --------------: | ----------------------------------- |
| 1    | `messy_invoice.png` (noisy, skewed)              | sparse |           18–19 |                      16–17 |    94.75–94.88% | PASS                                |
| 2    | `messy_invoice.png` (same input, block mode)     | block  |              20 |                         17 |          93.41% | PASS                                |
| 3    | `book_page.png` (4-line paragraph, noise + skew) | block  |              28 |                         28 |          94.39% | PASS                                |
| 4    | Missing/invalid file path                        | —      |               — |                          — |               — | Clean error (exit code 1, no crash) |

### Recognized Text Sample

Test 1 successfully recognized the following text:

```text
INVOICE #0042 DATE: 2026-08-22 ITEM: SERVER RACK UNIT 3 TOTAL: $499.00 DECODELABS AI TRAINING KIT STATUS: PAID
```

> **Note:** The exact token counts may vary slightly between test runs (for example, 16 vs. 17 accepted tokens) because `generate_sample.py` generates new random noise each time. This simulates real-world camera/scan noise, while the confidence gate consistently passes the test.

---

## PDF Requirements Mapping — Gatekeeper Rule (Slide 17)

| PDF Requirement                                                     | Implementation                                                                                                                           |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Library Integration** — Error-free `pytesseract` usage         | `run_ocr()` uses `pytesseract.image_to_data()` to extract per-token text, confidence scores, and bounding boxes                          |
| **2. Pre-Processing Integrity** — Grayscale + adaptive thresholding | `to_grayscale()` → `denoise()` → `deskew()` → `adaptive_threshold()` using true local/Gaussian adaptive thresholding for uneven lighting |
| **3. Accuracy Benchmarking** — Minimum 80% confidence               | `CONFIDENCE_THRESHOLD = 80` + `filter_by_confidence()`                                                                                   |
| **4. Visual Confirmation** — Legible output with boxes/labels       | `draw_confirmation()` — accepted text is marked with green boxes, rejected text with red boxes                                           |

The script also uses exit codes according to the validation gate:

* `0` → Gate passed
* `1` → Invalid or missing input file
* `2` → No token meets the 80% confidence threshold

This makes the OCR pipeline suitable for automated validation and CI environments.

---
