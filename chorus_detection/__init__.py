"""
chorus-detection: A tool for detecting chorus sections in audio files.

Quick start
-----------
    from chorus_detection.core.audio_processor import process_audio
    from chorus_detection.core.model import load_CRNN_model, make_predictions

    processed, features = process_audio("song.mp3")
    model = load_CRNN_model()
    predictions, starts, ends = make_predictions(model, processed, features)
"""

__version__ = "0.1.0"
__author__ = "Dennis Dang"

from chorus_detection.core.audio_processor import process_audio
from chorus_detection.core.model import load_CRNN_model, make_predictions, MODEL_PATH
from chorus_detection.core.wrapper import ChorusDetectionModel

__all__ = [
    "process_audio",
    "load_CRNN_model",
    "make_predictions",
    "MODEL_PATH",
    "ChorusDetectionModel",
]
