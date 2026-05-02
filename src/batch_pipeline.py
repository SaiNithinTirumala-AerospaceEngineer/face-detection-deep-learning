"""
batch_pipeline.py
-----------------
Batch image processing pipeline for face detection.

Processes an entire folder of images through the FaceDetector engine,
generates annotated outputs, produces a structured JSON report, and
prints a formatted summary table. Supports filtering by minimum face
count and custom output directories.

Extends face_detector.py with:
  - Folder-level batch processing with progress tracking
  - Per-image detection metadata export (JSON)
  - Summary statistics across the full batch
  - Detection rate, average face count, mean processing time
  - Failed image tracking and error reporting

Inputs : --input  folder of images (default: data/sample_images/)
         --output folder for annotated results (default: results/annotated/)
         --report path for JSON report (default: data/detection_results.json)
Outputs: annotated images in output folder
         JSON detection report
         printed summary table + batch statistics

Usage:
    # Default — process sample images
    python src/batch_pipeline.py

    # Custom input folder
    python src/batch_pipeline.py --input path/to/images/

    # With minimum face filter
    python src/batch_pipeline.py --min-faces 2

    # Quiet mode — no per-image output
    python src/batch_pipeline.py --quiet
"""

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np

# ── Add src to path so we can import face_detector ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from face_detector import FaceDetector

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_INPUT   = os.path.join(ROOT_DIR, "data", "sample_images")
DEFAULT_OUTPUT  = os.path.join(ROOT_DIR, "results", "annotated")
DEFAULT_REPORT  = os.path.join(ROOT_DIR, "data", "detection_results.json")

SUPPORTED_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Numpy type encoder for JSON ───────────────────────────────────────────────
def np_encode(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):   return bool(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


# ── Batch pipeline ────────────────────────────────────────────────────────────

def collect_images(input_dir):
    """Return sorted list of image paths from input directory."""
    images = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXT:
            images.append(os.path.join(input_dir, fname))
    return images


def process_batch(image_paths, detector, output_dir,
                  min_faces=0, quiet=False):
    """
    Run detection on all images in the batch.

    Parameters
    ----------
    image_paths : list of str
    detector    : FaceDetector instance
    output_dir  : str — where annotated images are saved
    min_faces   : int — only save annotated image if face_count >= min_faces
    quiet       : bool — suppress per-image output

    Returns
    -------
    dict: {filename: detection_result, ...}
    list: failed filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    failed  = []
    total   = len(image_paths)

    if not quiet:
        print(f"\n  {'#':<4} {'Filename':<32} {'Faces':>6} {'Eyes':>5} "
              f"{'Smile':>6} {'Coverage':>10} {'Time ms':>9}")
        print("  " + "─" * 75)

    for i, path in enumerate(image_paths, 1):
        fname = os.path.basename(path)

        image = cv2.imread(path)
        if image is None:
            failed.append(fname)
            if not quiet:
                print(f"  {i:<4} {fname:<32}  ERROR: could not load")
            continue

        # Detect
        detection = detector.detect(image)
        annotated = detector.annotate(image, detection)

        # Save annotated image if it meets the min_faces threshold
        if detection["face_count"] >= min_faces:
            stem    = os.path.splitext(fname)[0]
            out_path = os.path.join(output_dir, f"{stem}_annotated.jpg")
            cv2.imwrite(out_path, annotated)

        # Store result
        results[fname] = detection

        if not quiet:
            total_eyes = sum(f["eye_count"] for f in detection["faces"])
            any_smile  = any(f["smile_detected"] for f in detection["faces"])
            print(f"  {i:<4} {fname:<32} {detection['face_count']:>6} "
                  f"{total_eyes:>5} {'Yes' if any_smile else 'No':>6} "
                  f"{detection['face_coverage']*100:>9.1f}% "
                  f"{detection['processing_ms']:>8.1f}")

    return results, failed


def compute_batch_stats(results):
    """Compute summary statistics across all detection results."""
    if not results:
        return {}

    face_counts    = [r["face_count"]    for r in results.values()]
    coverages      = [r["face_coverage"] for r in results.values()]
    proc_times     = [r["processing_ms"] for r in results.values()]
    images_with_faces = sum(1 for c in face_counts if c > 0)

    return {
        "total_images":        len(results),
        "images_with_faces":   images_with_faces,
        "detection_rate_pct":  round(images_with_faces / len(results) * 100, 1),
        "total_faces_detected": sum(face_counts),
        "mean_faces_per_image": round(sum(face_counts) / len(face_counts), 2),
        "max_faces_in_image":   max(face_counts),
        "mean_coverage_pct":    round(sum(coverages) / len(coverages) * 100, 2),
        "mean_processing_ms":   round(sum(proc_times) / len(proc_times), 1),
        "total_processing_ms":  round(sum(proc_times), 1),
    }


def print_batch_summary(stats, failed):
    """Print formatted batch statistics."""
    print("\n  " + "═" * 55)
    print("  BATCH SUMMARY")
    print("  " + "═" * 55)
    print(f"  Images processed       : {stats['total_images']}")
    print(f"  Detection rate         : {stats['detection_rate_pct']}%  "
          f"({stats['images_with_faces']}/{stats['total_images']} images)")
    print(f"  Total faces detected   : {stats['total_faces_detected']}")
    print(f"  Mean faces per image   : {stats['mean_faces_per_image']:.2f}")
    print(f"  Max faces in one image : {stats['max_faces_in_image']}")
    print(f"  Mean face coverage     : {stats['mean_coverage_pct']:.2f}%")
    print(f"  Mean processing time   : {stats['mean_processing_ms']:.1f} ms")
    print(f"  Total processing time  : {stats['total_processing_ms']:.1f} ms")

    if failed:
        print(f"\n  Failed images ({len(failed)}):")
        for f in failed:
            print(f"    ✗ {f}")
    print("  " + "═" * 55)


def save_report(results, stats, failed, report_path):
    """Save full detection report as JSON."""
    report = {
        "batch_statistics": stats,
        "failed_images":    failed,
        "per_image_results": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=np_encode)
    print(f"\n  Report saved : {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch face detection pipeline")
    parser.add_argument("--input",      default=DEFAULT_INPUT,
                        help="Input folder containing images")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT,
                        help="Output folder for annotated images")
    parser.add_argument("--report",     default=DEFAULT_REPORT,
                        help="Path for JSON detection report")
    parser.add_argument("--min-faces",  type=int, default=0,
                        help="Only save annotated image if face_count >= N")
    parser.add_argument("--scale",      type=float, default=1.15,
                        help="Detector scaleFactor (default: 1.15)")
    parser.add_argument("--neighbours", type=int, default=5,
                        help="Detector minNeighbors (default: 5)")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress per-image output")
    args = parser.parse_args()

    print("Face Detection — Batch Pipeline")
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Report : {args.report}")
    print(f"  Params : scaleFactor={args.scale}, "
          f"minNeighbors={args.neighbours}, min_faces={args.min_faces}")

    # Collect images
    image_paths = collect_images(args.input)
    if not image_paths:
        print(f"\n  No images found in {args.input}")
        return
    print(f"\n  Found {len(image_paths)} image(s) to process")

    # Run batch
    t_start  = time.time()
    detector = FaceDetector(args.scale, args.neighbours)
    results, failed = process_batch(
        image_paths, detector, args.output,
        min_faces=args.min_faces, quiet=args.quiet
    )
    t_total = (time.time() - t_start) * 1000

    # Stats and report
    stats = compute_batch_stats(results)
    print_batch_summary(stats, failed)
    save_report(results, stats, failed, args.report)

    print(f"  Annotated images : {args.output}")
    print(f"  Wall-clock time  : {t_total:.0f} ms")


if __name__ == "__main__":
    main()