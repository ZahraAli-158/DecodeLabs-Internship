# Project 4 — Image/Text Recognition (Basic)
**DecodeLabs AI Industrial Training, Batch 2026 — Optional Mastery Phase**

Ye project PDF mei diye gaye **Path 1: OCR** requirements ko fully implement karta hai using `pytesseract` (Tesseract engine). Maine khud test kar liya hai — pura pipeline working hai (test results neeche diye hain).

## Files
```
ocr_project/
├── generate_sample.py     # synthetic "messy" invoice image generator (noise+shadow+skew)
├── ocr_pipeline.py         # MAIN script — ye hi Project 4 ka deliverable hai
├── sample_input/
│   ├── messy_invoice.png   # sample 1: noisy/skewed invoice
│   └── book_page.png       # sample 2: paragraph text (robustness test)
└── outputs/                 # har run ke baad results yahan bante hain
    ├── *_1_binary.png       # pre-processed (deskewed + adaptive thresholded) image
    ├── *_2_confirmed.png    # visual confirmation: green box = accepted, red = rejected
    ├── *_extracted.txt      # final extracted text
    └── *_report.json        # full machine-readable report
```

---

## Local machine pe setup kaise karna hai

### Step 1 — Tesseract OCR engine install karo (system-level, Python se alag)
- **Windows:** UB-Mannheim ka Tesseract installer download karo (`https://github.com/UB-Mannheim/tesseract/wiki`), run karo. Default path: `C:\Program Files\Tesseract-OCR`. Agar PATH mei auto-add nahi hota, `ocr_pipeline.py` ke sabse upar ye line add kar dena:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`

### Step 2 — Python libraries install karo
Project folder ke andar terminal/CMD khol ke:
```bash
pip install opencv-python pytesseract numpy pillow
```

### Step 3 — Sanity check (bundled sample pe)
```bash
python ocr_pipeline.py
```
Agar sab sahi hai toh terminal mei report print hoga aur last line `[GATE] PASSED` dikhegi. Agar `tesseract is not installed or it's not in your PATH` jaisi error aaye, Step 1 dobara dekho.

### Step 4 — Apni khud ki image pe run karo
```bash
python ocr_pipeline.py path/to/your_image.png --psm sparse
```
`--psm` options: `auto` (mixed layout), `block` (paragraph), `line` (single line — number plates/headers), `sparse` (invoices, scattered text).

### Step 5 — Output verify karo
`outputs/` folder mei 4 files banti hain har run pe — `*_2_confirmed.png` sabse zaroori hai defense ke liye (visually dikhata hai ke 80% gate kaam kar raha hai).

---

## Maine khud test kiya — results

| Test | Input | PSM | Tokens detected | Tokens **accepted (≥80%)** | Avg confidence | Gate |
|---|---|---|---|---|---|---|
| 1 | messy_invoice.png (noisy, skewed) | sparse | 18–19 | 16–17 | 94.75–94.88% | PASS |
| 2 | messy_invoice.png (same, block mode) | block | 20 | 17 | 93.41% | PASS |
| 3 | book_page.png (4-line paragraph, noise+skew) | block | 28 | 28 | 94.39% | PASS |
| 4 | missing/invalid file path | — | — | — | — | Clean error (exit code 1, no crash) |

Recognized text sample (Test 1): `INVOICE #0042 DATE: 2026-08-22 ITEM: SERVER RACK UNIT 3 TOTAL: $499.00 DECODELABS AI TRAINING KIT STATUS: PAID`

Note: Test runs mei numbers thoda vary karte hain (16 vs 17 tokens) kyunki har baar naya random noise generate hota hai (`generate_sample.py` mei) — ye real-world camera/scan noise jaisa hi behavior hai, aur gate har baar consistently pass hua.

## PDF requirements ka mapping (Gatekeeper Rule, slide 17)

| PDF Requirement | Kahan implement hua |
|---|---|
| **1. Library Integration** — error-free pytesseract usage | `run_ocr()` — `pytesseract.image_to_data()` se per-token text + confidence + bounding box |
| **2. Pre-Processing Integrity** — grayscale + adaptive thresholding | `to_grayscale()` → `denoise()` → `deskew()` → `adaptive_threshold()` (true local/Gaussian adaptive, isliye uneven lighting mei bhi kaam karta hai) |
| **3. Accuracy Benchmarking** — min 80% confidence | `CONFIDENCE_THRESHOLD = 80` + `filter_by_confidence()` |
| **4. Visual Confirmation** — legible output with boxes/labels | `draw_confirmation()` — accepted = green box + label, rejected = red box |

Script ka exit code bhi gate ke hisaab se set hota hai (`0` = pass, `1` = bad input file, `2` = koi bhi token 80% clear na kare), taake ye kisi automated check/CI mei bhi drop-in ho sake.

## Path 2 (Object Detection) — kyun nahi banaya
PDF mei clearly "OCR **or** Object Detection" likha tha (dono zaroori nahi). Path 1 pehle se hi sab 4 gates pass kar raha hai, isliye requirement fully complete hai. Path 2 (MobileNet-SSD) ke liye pre-trained `.caffemodel` weights internet se download karni padtin — agar bonus ke liye chahiye ho apne local machine pe, structure bata dena, code bana dungi.
