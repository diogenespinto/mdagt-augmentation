# MDAGT - Manual Data Augmentation via Geometric Transformations

This repository contains the code for the data augmentation pipeline used in:

"Automatic segmentation of the ribeye area in *Bos indicus* and *Bos taurus* cattle enhanced by symmetry and homothety-based data augmentation"

*Journal of Animal Science, 2026*

## Description

MDAGT generates 36 augmented variations from a single image-LabelMe JSON pair using three geometric transformations:

1. Homothety (scaling) - 3 levels: 0.95, 1.00, 1.05
2. Rotation - 3 angles: -5°, 0°, +5°
3. Symmetry (flips) - 4 types: none, horizontal, vertical, both

Total: 3 × 3 × 4 = 36 augmented samples per original image.

## Requirements

- Python 3.10+
- OpenCV 4.9+
- NumPy 1.26+

## Installation

```bash
pip install -r requirements.txt
