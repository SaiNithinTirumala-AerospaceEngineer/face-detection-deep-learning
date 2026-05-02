# IIT Hyderabad AI & Deep Learning Workshop

## Workshop details

| Field | Details |
|---|---|
| Institution | Indian Institute of Technology Hyderabad (IIT-H) |
| Organiser | ISIE India (www.isieindia.com) |
| Dates | 08–09 April 2023 |
| Duration | 2 days, morning + afternoon sessions |
| Format | Lecture + hands-on Python programming |

---

## Workshop curriculum

### Day 1 — AI Foundations and Python Programming

**Morning session:**
- Traditional programming vs AI programming paradigm
- Machine learning: 5-step pipeline (data acquisition, preprocessing,
  training, validation, deployment)
- Deep learning: neural network structure and applications
- Introduction to LFW (Labeled Faces in the Wild) dataset
- Roboflow platform for detection applications

**Afternoon session:**
- Python fundamentals: data types, operators, control statements
- Hands-on: first Python programs and debugging

### Day 2 — Scientific Python and ML Concepts

**Morning session:**
- Object-Oriented Programming (classes, inheritance, encapsulation)
- NumPy: array operations, linspace, broadcasting
- Pandas: dataframes, data cleaning, statistical analysis

**Afternoon session:**
- Matplotlib: line plots, scatter plots, bar charts, pie charts
- Regression vs classification
- Dependent vs independent variables
- Multiple linear regression, `fit()` function

---

## Connection to this project

The workshop provided the Python and AI/ML foundations. This repository
extends that foundation into a practical face detection implementation:

| Workshop concept | Applied in this project |
|---|---|
| AI programming paradigm | Pre-trained cascade classifiers — no manual rules |
| 5-step ML pipeline | Detection → preprocessing → inference → annotation → analysis |
| LFW dataset (mentioned Day 1) | Referenced in literature — cascade classifiers trained on LFW-style data |
| NumPy | Array operations throughout all scripts |
| Pandas | Data loading and batch statistics |
| Matplotlib | All 7 result plots |
| Python OOP | `FaceDetector` class encapsulating detection logic |

The parameter benchmark study (`parameter_benchmark.py`) directly
applies the workshop's AI programming concepts — using data-driven
optimisation to find the best detector configuration, rather than
manually guessing parameter values.