# Methodology — Face Detection using AI & Deep Learning

## Overview

This project implements a multi-feature face detection pipeline using
OpenCV's Haar cascade classifiers, extended with a systematic parameter
sensitivity study and batch processing infrastructure. The work was
developed as a practical extension of the AI & Deep Learning Workshop
held at IIT Hyderabad (08–09 April 2023).

---

## Algorithm — Viola-Jones Haar Cascade

### Background

The Haar cascade classifier was introduced by Viola and Jones (2001)
and remains a foundational algorithm in computer vision. It operates
on greyscale images using Haar-like features — rectangular regions
where the difference in pixel intensity between adjacent areas is used
to detect structural patterns characteristic of human faces.

The algorithm consists of three key innovations:

**1. Haar-like features**
Rectangular feature filters are applied across the image at multiple
scales. For face detection, the most discriminative features capture:
- Dark eye regions flanking a bright nose bridge
- Bright forehead above dark eye regions
- Dark mouth region below bright cheeks

**2. Integral image**
A summed area table is pre-computed once per image, allowing any
rectangular region sum to be computed in constant time (O(1)),
making the feature computation fast enough for real-time use.

**3. AdaBoost cascade**
Weak classifiers (single Haar features) are combined using AdaBoost
into a strong classifier. The cascade structure means most image
regions are rejected early — only regions passing all stages are
classified as faces. This gives the algorithm its speed advantage.

### Cascade classifiers used

| Classifier | File | Detects |
|---|---|---|
| Face | `haarcascade_frontalface_default.xml` | Frontal faces |
| Eye | `haarcascade_eye.xml` | Eyes within face ROI |
| Smile | `haarcascade_smile.xml` | Smile within lower face ROI |

All three classifiers are pre-trained and bundled with OpenCV.
They were trained on the LFW (Labeled Faces in the Wild) dataset
and similar large-scale face image corpora.

---

## Preprocessing pipeline

```
Input image (BGR)
      │
      ▼
Greyscale conversion — cv2.cvtColor(img, COLOR_BGR2GRAY)
      │
      ▼
Histogram equalisation — cv2.equalizeHist(grey)
      │    (redistributes pixel intensities for improved contrast)
      │    (critical for low-light and high-contrast images)
      ▼
Face detection — detectMultiScale(grey, scaleFactor, minNeighbors)
      │
      ▼
Per-face ROI extraction
      │
      ├── Eye detection — full face ROI
      └── Smile detection — lower 50% of face ROI
```

Histogram equalisation is applied before detection because the cascade
classifiers were trained on images with roughly uniform intensity
distributions. Without equalisation, dark images (like test6_dark_bg)
produce significantly fewer detections.

---

## Key parameters

| Parameter | Value used | Effect |
|---|---|---|
| `scaleFactor` | 1.15 | Controls image pyramid — lower = more detections, slower |
| `minNeighbors` | 5 | Minimum overlapping detections required — higher = fewer false positives |
| `minSize` | (30, 30) | Minimum face size in pixels |

### Parameter benchmark findings

The parameter sensitivity study (56 combinations) identified:

**Optimal operating point: scaleFactor=1.10, minNeighbors=4**
- 8 true positives across 6 images
- 0 false positives
- Mean processing time: ~192 ms per image

**Speed-accuracy trade-off:**
- scaleFactor=1.05 → 4× slower than 1.40, but detects small/partial faces
- scaleFactor=1.40 → fastest (45–95 ms) but misses faces below ~100×100 px
- minNeighbors=2 → highest recall (most faces found), most false positives
- minNeighbors=8 → highest precision (all detections correct), misses more faces

---

## Detection results

| Image | Faces (detected) | Faces (expected) | Match |
|---|---|---|---|
| test1_single_face | 1 | 1 | ✓ |
| test2_two_faces | 2 | 2 | ✓ |
| test3_gradient_bg | 1 | 1 | ✓ |
| test4_group | 3 | 3 | ✓ |
| test5_small_face | 0 | 1 | ✗ |
| test6_dark_bg | 1 | 1 | ✓ |

**Exact match accuracy: 5/6 = 83.3%**

The missed detection (test5_small_face) occurs because the face
subtends less than 30×30 pixels — below the `minSize` threshold.
Reducing minSize to (15,15) with scaleFactor=1.05 recovers this
detection at the cost of ~2× processing time.

---

## Limitations and future scope

- Haar cascades are trained on frontal faces — profile and tilted faces
  (>30° rotation) are frequently missed. A DNN-based detector
  (ResNet-SSD or MTCNN) would handle non-frontal faces.
- Small faces (< 30px) are not detectable at default settings — a
  multi-scale approach with image upsampling is the standard extension.
- No tracking across video frames — a Kalman filter or DeepSORT tracker
  would provide stable face bounding boxes in webcam mode.
- Smile detection is sensitive to lighting and produces false positives
  in high-contrast images.

---

## References

- Viola, P. and Jones, M. (2001) Rapid Object Detection using a Boosted
  Cascade of Simple Features. *IEEE CVPR 2001*.
- Lienhart, R. and Maydt, J. (2002) An Extended Set of Haar-like Features
  for Rapid Object Detection. *IEEE ICIP 2002*.
- LFW: Labeled Faces in the Wild — http://vis-www.cs.umass.edu/lfw/
- OpenCV Haar cascade documentation — https://docs.opencv.org