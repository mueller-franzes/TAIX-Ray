"""Apply Detection+Recognition OCR (CRAFT + EasyOCR) to X-ray images and save results to CSV."""
import argparse
import csv
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    import cv2
except Exception as exc:
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as exc:
    np = None
    NP_IMPORT_ERROR = exc
else:
    NP_IMPORT_ERROR = None

try:
    import easyocr
except Exception as exc:
    easyocr = None
    EASYOCR_IMPORT_ERROR = exc
else:
    EASYOCR_IMPORT_ERROR = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _scandir_recursive(directory):
    """Fast recursive directory scan using os.scandir()."""
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    yield from _scandir_recursive(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in IMAGE_EXTS:
                        yield Path(entry.path)
    except PermissionError:
        pass


def _iter_images(root: Path, show_progress: bool = False):
    iterator = _scandir_recursive(str(root))
    if show_progress:
        iterator = tqdm(iterator, desc="Scanning for images", unit="img")
    yield from iterator


def _easyocr_detect(
    reader,
    image_bgr,
    text_min_confidence: float,
    box_min_confidence: float,
    det_params: dict,
):
    # EasyOCR uses CRAFT for detection internally.
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(image_rgb, detail=1, paragraph=False, **det_params)

    texts = []
    confs = []
    boxes = []
    total = 0

    for bbox, text, conf in results:
        total += 1
        if conf >= box_min_confidence:
            boxes.append(bbox)
        if not text or not text.strip():
            continue
        if conf < text_min_confidence:
            continue
        texts.append(text.strip())
        confs.append(float(conf))

    joined = " | ".join(texts).strip()
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return joined, avg_conf, boxes, total


def run_ocr(
    input_dir: Path,
    output_csv: Path,
    output_images_dir: Path | None,
    output_preprocessed_dir: Path | None,
    text_min_confidence: float,
    box_min_confidence: float,
    langs: list[str],
    use_gpu: bool,
    det_params: dict,
    verbose: bool,
):
    if cv2 is None:
        raise RuntimeError(f"opencv-python import failed: {CV2_IMPORT_ERROR}")
    if np is None:
        raise RuntimeError(f"numpy import failed: {NP_IMPORT_ERROR}")
    if easyocr is None:
        raise RuntimeError(f"easyocr import failed: {EASYOCR_IMPORT_ERROR}")

    reader = easyocr.Reader(langs, gpu=use_gpu)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_images_dir is not None:
        output_images_dir.mkdir(parents=True, exist_ok=True)
    if output_preprocessed_dir is not None:
        output_preprocessed_dir.mkdir(parents=True, exist_ok=True)

    images = list(_iter_images(input_dir, show_progress=True))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text", "confidence", "region"])

        for img_path in tqdm(images, desc="Processing images", unit="img"):
            try:
                if output_preprocessed_dir is not None:
                    dst = output_preprocessed_dir / img_path.name
                    if dst.resolve() != img_path.resolve():
                        shutil.copy2(img_path, dst)

                image = cv2.imread(str(img_path))
                if image is None:
                    raise RuntimeError("OpenCV failed to read image")

                text, conf, boxes, total = _easyocr_detect(
                    reader,
                    image,
                    text_min_confidence=text_min_confidence,
                    box_min_confidence=box_min_confidence,
                    det_params=det_params,
                )
                if verbose:
                    print(f"{img_path.name}: detections={total} boxes_drawn={len(boxes)}")

                if not text:
                    continue

                writer.writerow([img_path.name, text, f"{conf:.2f}", "full"])

                if output_images_dir is not None and boxes:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    viz = Image.fromarray(rgb)
                    draw = ImageDraw.Draw(viz)
                    for box in boxes:
                        pts = [(int(x), int(y)) for x, y in box]
                        if len(pts) >= 2:
                            draw.line(pts + [pts[0]], fill="red", width=2)
                    viz.save(output_images_dir / img_path.name)

            except Exception as exc:
                writer.writerow([img_path.name, f"[ERROR] {exc}", "", ""])


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR over X-ray images using CRAFT (via EasyOCR) + EasyOCR recognition."
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/ocean_storage/data/TAIX/download/data",
        help="Folder containing images. Default: test_images_with_text",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/ocr_texts.csv",
        help="Output CSV path. Default: outputs/ocr_texts.csv",
    )
    parser.add_argument(
        "--output-images-dir",
        default="outputs/ocr_boxes",
        help="Directory to write annotated images. Default: outputs/ocr_boxes",
    )
    parser.add_argument(
        "--output-preprocessed-dir",
        default="",
        help="Directory to write OCR input images. Default: disabled",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.01,
        help="Minimum OCR confidence (0-1) to keep text. Default: 0.3.",
    )
    parser.add_argument(
        "--box-min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence (0-1) to draw a box. Default: 0.0.",
    )
    parser.add_argument(
        "--langs",
        default="en",
        help="Comma-separated EasyOCR language codes. Default: en",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Deprecated alias for --langs (single language code).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU and use CPU instead.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=None,
        help="EasyOCR text_threshold (lower = more boxes). Default: use library default.",
    )
    parser.add_argument(
        "--low-text",
        type=float,
        default=None,
        help="EasyOCR low_text (lower = more boxes). Default: use library default.",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=None,
        help="EasyOCR link_threshold (lower = more boxes). Default: use library default.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detection counts per image.",
    )

    args = parser.parse_args()
    if args.lang:
        langs = [args.lang]
    else:
        langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    det_params = {}
    if args.text_threshold is not None:
        det_params["text_threshold"] = args.text_threshold
    if args.low_text is not None:
        det_params["low_text"] = args.low_text
    if args.link_threshold is not None:
        det_params["link_threshold"] = args.link_threshold

    run_ocr(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        output_images_dir=Path(args.output_images_dir) if args.output_images_dir else None,
        output_preprocessed_dir=Path(args.output_preprocessed_dir)
        if args.output_preprocessed_dir
        else None,
        text_min_confidence=args.min_confidence,
        box_min_confidence=args.box_min_confidence,
        langs=langs,
        use_gpu=not args.no_gpu,
        det_params=det_params,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
