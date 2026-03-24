# Automated Chorus Detection - Package
This is a package built from this [repository](https://github.com/dennisvdang/chorus-detection). For transparency while the original code is not vibe-coded, this is.

## Quick Installation

```bash
git clone https://github.com/anklebowl/chorus-detection.git
cd chorus-detection
pip install .
```

```bash
chorus-detection --file song.mp3
```

```python
from chorus_detection import ChorusDetectionModel
import librosa

model = ChorusDetectionModel()

audio, sr = librosa.load(AUDIO_PATH)
out = model.predict(audio, sr)
```

# Automated Chorus Detection [![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/dennisvdang/chorus-detection)

![Chorus Prediction](./images/131.webp)

## Overview

A hierarchical convolutional recurrent neural network designed for musical structure analysis, specifically optimized for detecting choruses in music recordings. The model was initially trained on 332 annotated songs from electronic music genres and achieved an F1 score of 0.864 (Precision: 0.831, Recall: 0.900) on unseen test data. For more details, scroll down to the [Project Technical Summary section](#project-technical-summary).

## Quick Links

- [Try the model on HuggingFace Spaces](https://huggingface.co/spaces/dennisvdang/Chorus-Detection)
- [Labeled training dataset of 332 songs (audio files not included)](data/clean_labeled.csv)
- [Pre-trained model file](chorus_detection/models/CRNN/best_model_V3.h5)
- [Model training notebook](notebooks/Automated-Chorus-Detection.ipynb)
- [Music annotation process](docs/Data_Annotation_Guide.pdf)
- [Project PDF writeup](docs/Capstone_Final_Report.pdf)

## Project Structure

```
chorus-detection/
├── chorus_detection/        # Installable package
│   ├── core/                # Core functionality
│   │   ├── audio_processor.py   # Audio processing and feature extraction
│   │   ├── model.py             # Model loading and prediction
│   │   ├── utils.py             # Utility functions
│   │   └── visualization.py     # Plotting and visualization
│   ├── cli/                 # Command-line interface
│   │   └── cli_app.py           # CLI application
│   └── models/              # Pre-trained models
├── pyproject.toml           # Package metadata and build config
├── setup.py                 # Legacy build shim
└── requirements.txt         # Runtime dependencies
```

## Project Technical Summary

Technical information about this project is available [here](https://github.com/dennisvdang/chorus-detection)