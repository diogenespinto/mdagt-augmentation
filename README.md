# MDAGT - Manual Data Augmentation via Geometric Transformations

This repository contains the code for the data augmentation pipeline used in:

> "Automatic segmentation of the ribeye area in Bos indicus and Bos taurus cattle enhanced by symmetry and homothety-based data augmentation"

Journal of Animal Science, 2026

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

pip install -r requirements.txt

## Usage

1. Place your image and LabelMe JSON file in the same directory
2. Update the paths in mdagt_pipeline.py:

- INPUT_IMAGE = "./BTT.jpg" # or "./BTI.png"
- INPUT_JSON = "./BTT.json" # must match the image name
- OUTPUT_DIR = "./mdagt_augmented"

3. Run the script:

python mdagt_pipeline.py

## Output

The script generates:
- 36 augmented images (JPG format)
- 36 corresponding LabelMe JSON files

## Citation

If you use this code in your research, please cite:

[Your article citation]

## License

MIT License
