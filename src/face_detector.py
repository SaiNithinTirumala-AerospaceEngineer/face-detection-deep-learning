"""
face_detector.py
----------------
Core face detection engine using OpenCV Haar cascade classifiers.

Detects faces, eyes (within each face region), and smiles using
pre-trained cascade classifiers bundled with OpenCV. Supports
single image detection and real-time webcam mode.

Detection pipeline:
    1. Load image → convert to greyscale
    2. Histogram equalisation → improve contrast for detection
    3. Face detection (Haar cascade, frontal face)
    4. Per-face: eye detection + smile detection
    5. Annotate image with colour-coded bounding boxes
    6. Return structured detection dictionary

Cascade classifiers used:
    - haarcascade_frontalface_default.xml  (face)
    - haarcascade_eye.xml                  (eyes)
    - haarcascade_smile.xml                (smile)

Source: Viola, P. and Jones, M. (2001) Rapid Object Detection using
a Boosted Cascade of Simple Features. IEEE CVPR.

Inputs : image file path or webcam index
Outputs: annotated image, detection dictionary

Usage:
    # Single image
    python src/face_detector.py --image data/sample_images/test1_single_face.jpg

    # All sample images
    python src/face_detector.py --all

    # Webcam (press Q to quit)
    python src/face_detector.py --webcam
"""

import cv2
import numpy as np
import os
import json
import argparse
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(__file__)
ROOT_DIR    = os.path.join(SRC_DIR, "..")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "annotated")
DATA_DIR    = os.path.join(ROOT_DIR, "data", "sample_images")

# ── Cascade classifier paths (bundled with OpenCV) ────────────────────────────
CASCADE_FACE  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_EYE   = cv2.data.haarcascades + "haarcascade_eye.xml"
CASCADE_SMILE = cv2.data.haarcascades + "haarcascade_smile.xml"

# ── Detection parameters ──────────────────────────────────────────────────────
FACE_SCALE_FACTOR   = 1.15   # Image pyramid scale — lower = more detections
FACE_MIN_NEIGHBOURS = 5      # Minimum neighbours to retain detection
FACE_MIN_SIZE       = (30, 30)

EYE_SCALE_FACTOR    = 1.1
EYE_MIN_NEIGHBOURS  = 10
EYE_MIN_SIZE        = (20, 20)

SMILE_SCALE_FACTOR   = 1.7
SMILE_MIN_NEIGHBOURS = 22
SMILE_MIN_SIZE       = (25, 25)

# ── Annotation colours (BGR) ──────────────────────────────────────────────────
COLOUR_FACE  = (0, 200, 0)    # Green
COLOUR_EYE   = (255, 100, 0)  # Blue
COLOUR_SMILE = (0, 100, 255)  # Red
COLOUR_LABEL = (255, 255, 255)
COLOUR_BG    = (0, 0, 0)


# ── Core detection class ──────────────────────────────────────────────────────

class FaceDetector:
    """
    Multi-feature face detector using OpenCV Haar cascade classifiers.

    Detects faces, eyes per face, and smiles per face from images or
    webcam feed. Returns structured detection results with bounding
    box coordinates and feature counts.
    """

    def __init__(self,
                 face_scale=FACE_SCALE_FACTOR,
                 face_min_neighbours=FACE_MIN_NEIGHBOURS):
        self.face_cascade  = cv2.CascadeClassifier(CASCADE_FACE)
        self.eye_cascade   = cv2.CascadeClassifier(CASCADE_EYE)
        self.smile_cascade = cv2.CascadeClassifier(CASCADE_SMILE)

        self.face_scale         = face_scale
        self.face_min_neighbours = face_min_neighbours

        if self.face_cascade.empty():
            raise RuntimeError(
                "Face cascade failed to load. Check OpenCV installation.")

    def detect(self, image):
        """
        Run full detection pipeline on a BGR image array.

        Parameters
        ----------
        image : np.ndarray
            BGR image (as loaded by cv2.imread)

        Returns
        -------
        dict with keys:
            face_count      : int
            faces           : list of dicts with x,y,w,h,eyes,smile
            image_area_px   : int
            face_area_px    : int
            face_coverage   : float (fraction of image covered by faces)
            processing_ms   : float
        """
        t_start = time.time()

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)

        raw_faces = self.face_cascade.detectMultiScale(
            grey,
            scaleFactor=self.face_scale,
            minNeighbors=self.face_min_neighbours,
            minSize=FACE_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        results      = []
        total_face_area = 0

        if len(raw_faces) > 0:
            for (x, y, w, h) in raw_faces:
                face_grey = grey[y:y+h, x:x+w]

                # Eye detection within face ROI
                eyes = self.eye_cascade.detectMultiScale(
                    face_grey,
                    scaleFactor=EYE_SCALE_FACTOR,
                    minNeighbors=EYE_MIN_NEIGHBOURS,
                    minSize=EYE_MIN_SIZE
                )

                # Smile detection within lower half of face ROI
                face_lower = face_grey[h//2:, :]
                smiles = self.smile_cascade.detectMultiScale(
                    face_lower,
                    scaleFactor=SMILE_SCALE_FACTOR,
                    minNeighbors=SMILE_MIN_NEIGHBOURS,
                    minSize=SMILE_MIN_SIZE
                )

                total_face_area += w * h

                results.append({
                    "x": int(x), "y": int(y),
                    "w": int(w), "h": int(h),
                    "area_px": int(w * h),
                    "eye_count":   int(len(eyes)),
                    "smile_detected": bool(len(smiles) > 0),
                    "eyes": [{"x": int(ex), "y": int(ey),
                               "w": int(ew), "h": int(eh)}
                              for (ex, ey, ew, eh) in eyes],
                })

        h_img, w_img = image.shape[:2]
        image_area   = h_img * w_img
        processing_ms = (time.time() - t_start) * 1000

        return {
            "face_count":    len(results),
            "faces":         results,
            "image_area_px": image_area,
            "face_area_px":  total_face_area,
            "face_coverage": total_face_area / image_area if image_area > 0 else 0,
            "processing_ms": round(processing_ms, 2),
        }

    def annotate(self, image, detection):
        """
        Draw bounding boxes and labels on a copy of the image.

        Returns annotated BGR image.
        """
        annotated = image.copy()

        for i, face in enumerate(detection["faces"]):
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]

            # Face rectangle
            cv2.rectangle(annotated, (x, y), (x+w, y+h), COLOUR_FACE, 2)

            # Label background + text
            label = f"Face {i+1}"
            if face["smile_detected"]:
                label += " :)"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated,
                          (x, y-lh-8), (x+lw+4, y), COLOUR_FACE, -1)
            cv2.putText(annotated, label, (x+2, y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_LABEL, 1)

            # Eye rectangles (coordinates relative to face ROI)
            face_roi = annotated[y:y+h, x:x+w]
            for eye in face["eyes"]:
                ex, ey, ew, eh = eye["x"], eye["y"], eye["w"], eye["h"]
                cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh),
                              COLOUR_EYE, 1)

        # Summary overlay
        summary = (f"Faces: {detection['face_count']}  "
                   f"Coverage: {detection['face_coverage']*100:.1f}%  "
                   f"Time: {detection['processing_ms']:.0f}ms")
        cv2.rectangle(annotated, (0, 0), (len(summary)*9, 25),
                      (40, 40, 40), -1)
        cv2.putText(annotated, summary, (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_LABEL, 1)

        return annotated


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_single(image_path, detector, save=True):
    """Detect faces in a single image and optionally save annotated result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ERROR: Could not load image: {image_path}")
        return None

    detection = detector.detect(image)
    annotated = detector.annotate(image, detection)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fname   = os.path.basename(image_path)
        out_path = os.path.join(RESULTS_DIR,
                                fname.replace(".", "_annotated."))
        cv2.imwrite(out_path, annotated)

    return detection, annotated


def run_all_samples(detector):
    """Run detection on all images in data/sample_images/."""
    images = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\nRunning detection on {len(images)} sample images...\n")
    print(f"  {'Image':<35} {'Faces':>6} {'Eyes':>6} "
          f"{'Smile':>6} {'Coverage':>10} {'Time (ms)':>10}")
    print("  " + "─" * 75)

    all_results = {}
    for fname in images:
        path   = os.path.join(DATA_DIR, fname)
        out = run_single(path, detector, save=True)
        if out is None:
            continue
        result, _ = out

        total_eyes = sum(f["eye_count"] for f in result["faces"])
        any_smile  = any(f["smile_detected"] for f in result["faces"])

        print(f"  {fname:<35} {result['face_count']:>6} "
              f"{total_eyes:>6} {'Yes' if any_smile else 'No':>6} "
              f"{result['face_coverage']*100:>9.1f}% "
              f"{result['processing_ms']:>9.1f}")

        all_results[fname] = result

    # Save JSON report — custom encoder handles numpy int32/float32 types
    def _np_encode(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    report_path = os.path.join(ROOT_DIR, "data", "detection_results.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_np_encode)
    print(f"\n  Detection report saved: {report_path}")
    print(f"  Annotated images saved: {RESULTS_DIR}/")

    return all_results


def run_webcam(detector):
    """Real-time face detection from webcam. Press Q to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam mode — press Q to quit")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = detector.detect(frame)
        annotated = detector.annotate(frame, detection)

        cv2.imshow("Face Detection — Press Q to quit", annotated)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Webcam session ended. Processed {frame_count} frames.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face detection using OpenCV Haar cascade classifiers")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str,
                       help="Path to a single image file")
    group.add_argument("--all",    action="store_true",
                       help="Run on all sample images in data/sample_images/")
    group.add_argument("--webcam", action="store_true",
                       help="Run real-time detection from webcam")

    parser.add_argument("--scale", type=float, default=FACE_SCALE_FACTOR,
                        help=f"scaleFactor (default: {FACE_SCALE_FACTOR})")
    parser.add_argument("--neighbours", type=int,
                        default=FACE_MIN_NEIGHBOURS,
                        help=f"minNeighbors (default: {FACE_MIN_NEIGHBOURS})")
    args = parser.parse_args()

    detector = FaceDetector(
        face_scale=args.scale,
        face_min_neighbours=args.neighbours
    )

    print("Face Detection Engine — OpenCV Haar Cascade")
    print(f"  scaleFactor   : {args.scale}")
    print(f"  minNeighbors  : {args.neighbours}")

    if args.image:
        result, _ = run_single(args.image, detector, save=True)
        if result:
            print(f"\n  Faces detected : {result['face_count']}")
            print(f"  Face coverage  : {result['face_coverage']*100:.1f}%")
            print(f"  Processing time: {result['processing_ms']:.1f} ms")
            for i, face in enumerate(result["faces"]):
                print(f"\n  Face {i+1}:")
                print(f"    Bounding box : "
                      f"({face['x']},{face['y']}) "
                      f"{face['w']}×{face['h']} px")
                print(f"    Eyes detected: {face['eye_count']}")
                print(f"    Smile        : "
                      f"{'Yes' if face['smile_detected'] else 'No'}")

    elif args.all:
        run_all_samples(detector)

    elif args.webcam:
        run_webcam(detector)


if __name__ == "__main__":
    main()