"""Scan OCR results for PHI using a local LLM."""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

try:
    import openai
except ImportError:
    openai = None


def analyze_batch_for_phi(client, model: str, texts: list[str], max_retries: int = 3) -> list[dict]:
    """Analyze multiple texts in a single LLM call for efficiency."""
    if not texts:
        return []

    batch_content = "\n".join(
        f"{i+1}. \"{text}\""
        for i, text in enumerate(texts)
    )

    batch_prompt = f"""You are a medical data privacy expert. Analyze the following OCR texts extracted from X-ray images and determine if any contain Protected Health Information (PHI).

PHI includes:
- Patient names
- Dates (birth dates, admission dates, etc.)
- Medical record numbers or patient IDs
- Hospital/institution names combined with patient identifiers
- Addresses, phone numbers, email addresses
- Social Security numbers or other government IDs

Common NON-PHI text in X-rays (should NOT be flagged):
- Laterality markers: "R", "L", "RIGHT", "LEFT"
- Positioning: "sitzend", "liegend", "stehend", "AP", "PA", "lateral"
- Technical notes: "erschwert", "Aufnahme", "exspiration", "inspiration"
- Generic numbers without context
- Equipment/technique markers

Texts to analyze:
{batch_content}

Respond with JSON containing an array of results:
{{
  "results": [
    {{
      "index": 1,
      "contains_phi": true/false,
      "phi_type": "type of PHI or null",
      "explanation": "brief explanation",
      "confidence": "high/medium/low"
    }},
    ...
  ]
}}"""

    messages = [{"role": "user", "content": batch_prompt}]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=120,
            )
            content = response.choices[0].message.content
            return json.loads(content).get("results", [])
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"\n  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

    raise last_error


def find_image(filename: str, search_dirs: list[Path]) -> Path | None:
    """Find an image file in the search directories."""
    for search_dir in search_dirs:
        # Direct match
        candidate = search_dir / filename
        if candidate.exists():
            return candidate
        # Recursive search
        matches = list(search_dir.rglob(filename))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Scan OCR results for PHI using LLM")
    parser.add_argument(
        "--input-csv",
        default="outputs/ocr_texts.csv",
        help="Input CSV with OCR results. Default: outputs/ocr_texts.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/phi_findings.csv",
        help="Output CSV with PHI findings only. Default: outputs/phi_findings.csv",
    )
    parser.add_argument(
        "--output-images-dir",
        default="outputs/phi_images",
        help="Directory to copy images with PHI. Default: outputs/phi_images",
    )
    parser.add_argument(
        "--image-source-dir",
        default="/mnt/ocean_storage/data/TAIX/download/data",
        help="Directory where original images are stored.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of texts to analyze per LLM call. Default: 5",
    )
    args = parser.parse_args()

    if openai is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("GPTOSS_API_KEY")
    base_url = os.getenv("GPTOSS_BASE_URL")
    model = os.getenv("GPTOSS_MODEL")

    if not all([api_key, base_url, model]):
        raise RuntimeError("Missing environment variables: GPTOSS_API_KEY, GPTOSS_BASE_URL, GPTOSS_MODEL")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Read input CSV
    input_path = Path(args.input_csv)
    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} OCR results from {input_path}")

    # Prepare output directories
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_images_dir = Path(args.output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)

    image_source_dir = Path(args.image_source_dir)

    # Collect PHI findings
    phi_findings = []
    errors = []

    for i in tqdm(range(0, len(rows), args.batch_size), desc="Analyzing batches", unit="batch"):
        batch = rows[i:i + args.batch_size]
        texts = [row["text"] for row in batch]

        try:
            results = analyze_batch_for_phi(client, model, texts)

            # Map results back to original rows by index
            result_map = {r.get("index", idx + 1): r for idx, r in enumerate(results)}

            for j, row in enumerate(batch):
                result = result_map.get(j + 1, {})
                contains_phi = result.get("contains_phi", False)

                if contains_phi:
                    phi_findings.append({
                        "filename": row["filename"],
                        "text": row["text"],
                        "ocr_confidence": row["confidence"],
                        "phi_type": result.get("phi_type", ""),
                        "explanation": result.get("explanation", ""),
                        "llm_confidence": result.get("confidence", ""),
                    })

        except Exception as e:
            errors.append(f"Batch {i // args.batch_size + 1}: {e}")

    # Write PHI findings to CSV
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "text", "ocr_confidence", "phi_type", "explanation", "llm_confidence"])
        for finding in phi_findings:
            writer.writerow([
                finding["filename"],
                finding["text"],
                finding["ocr_confidence"],
                finding["phi_type"],
                finding["explanation"],
                finding["llm_confidence"],
            ])

    # Copy images with PHI to output directory
    images_copied = 0
    images_not_found = []
    for finding in phi_findings:
        img_path = find_image(finding["filename"], [image_source_dir])
        if img_path:
            dst = output_images_dir / finding["filename"]
            shutil.copy2(img_path, dst)
            images_copied += 1
        else:
            images_not_found.append(finding["filename"])

    # Print summary
    print("\n" + "=" * 60)
    print("PHI SCAN SUMMARY")
    print("=" * 60)
    print(f"Total images analyzed: {len(rows)}")
    print(f"PHI findings: {len(phi_findings)}")
    print(f"Images copied to {output_images_dir}: {images_copied}")
    if images_not_found:
        print(f"Images not found: {len(images_not_found)}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
    print("=" * 60)

    if phi_findings:
        print("\nPHI FINDINGS DETAIL:")
        print("-" * 60)
        for i, finding in enumerate(phi_findings, 1):
            print(f"\n{i}. {finding['filename']}")
            print(f"   Text: \"{finding['text']}\"")
            print(f"   Type: {finding['phi_type']}")
            print(f"   Explanation: {finding['explanation']}")
            print(f"   Confidence: {finding['llm_confidence']}")
    else:
        print("\nNo PHI detected in any images.")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
