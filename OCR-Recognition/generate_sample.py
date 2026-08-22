"""
generate_sample.py
-------------------
Creates a synthetic 'messy' sample document image so the OCR pipeline
has something realistic to prove itself on: clean text is rendered,
then degraded with Gaussian noise, uneven lighting, and a slight skew
-- exactly the kind of raw visual data described in Project 4
(shadows, chromatic noise, uneven lighting, tilted text lines).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

OUT_DIR = "sample_input"
os.makedirs(OUT_DIR, exist_ok=True)


def build_clean_document():
    """Render a clean 'invoice' as the ground-truth text source."""
    W, H = 900, 500
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
    except IOError:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    lines = [
        (40, 40, "INVOICE #0042", font_big),
        (40, 110, "DATE: 2026-08-22", font_small),
        (40, 150, "ITEM: SERVER RACK UNIT", font_small),
        (40, 190, "QTY: 3", font_small),
        (40, 230, "TOTAL: $499.00", font_small),
        (40, 300, "DECODELABS AI TRAINING KIT", font_small),
        (40, 340, "STATUS: PAID", font_small),
    ]
    for x, y, text, font in lines:
        draw.text((x, y), text, fill=(10, 10, 10), font=font)

    return np.array(img)


def degrade(img_rgb):
    """Apply noise, uneven lighting, and a slight rotation (skew)."""
    h, w = img_rgb.shape[:2]

    # 1. Uneven lighting: multiply by a soft gradient mask
    y_grad, x_grad = np.mgrid[0:h, 0:w]
    lighting = 0.55 + 0.45 * (x_grad / w)  # darker on the left
    lighting = lighting[..., None]
    lit = np.clip(img_rgb.astype(np.float32) * lighting, 0, 255)

    # 2. Gaussian / chromatic noise
    noise = np.random.normal(0, 12, lit.shape)
    noisy = np.clip(lit + noise, 0, 255).astype(np.uint8)

    # 3. Slight skew (tilt the whole page like a bad scan)
    angle = -4.5  # degrees
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(
        noisy, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )

    return skewed


if __name__ == "__main__":
    clean = build_clean_document()
    messy = degrade(clean)

    out_path = os.path.join(OUT_DIR, "messy_invoice.png")
    cv2.imwrite(out_path, cv2.cvtColor(messy, cv2.COLOR_RGB2BGR))
    print(f"Sample raw visual input generated -> {out_path}")
