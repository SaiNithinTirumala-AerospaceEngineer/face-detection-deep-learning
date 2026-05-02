# Face Detection using AI & Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Context](https://img.shields.io/badge/IIT%20Hyderabad-AI%20Workshop-orange)

## Problem statement

Manual image inspection for face detection is impractical at scale.
This project implements a multi-feature face detection pipeline using
OpenCV's Haar cascade classifiers — detecting faces, eyes, and smiles
in both static images and real-time webcam feeds. A parameter
benchmarking study quantifies the precision/recall trade-off across
detector settings, providing engineering insight beyond a standard
tutorial implementation.

*Developed as an extension of the AI & Deep Learning Workshop,
IIT Hyderabad, April 2023.*

---

## Repository under active development

Results, analysis scripts, and full documentation will be added in
subsequent commits.

---

## How to run

```bash
git clone https://github.com/SaiNithinTirumala-AerospaceEngineer/face-detection-deep-learning.git
cd face-detection-deep-learning
pip install -r requirements.txt

python src/face_detector.py --image data/sample_images/test1.jpg
python src/batch_pipeline.py --input data/sample_images/
python src/detection_analysis.py
python src/parameter_benchmark.py
```

---

## References

- Viola, P. and Jones, M. (2001) Rapid Object Detection using a
  Boosted Cascade of Simple Features. *IEEE CVPR*.
- LFW: Labeled Faces in the Wild — http://vis-www.cs.umass.edu/lfw/
- OpenCV Documentation — https://docs.opencv.org