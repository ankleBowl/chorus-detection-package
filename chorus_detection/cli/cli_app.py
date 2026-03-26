#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chorus Detection CLI Tool

This module provides a command-line interface for detecting choruses in audio files,
either from local audio files or YouTube URLs.
"""

import os
#
import sys
import argparse
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings("ignore")

from chorus_detection.core.audio_processor import process_audio
from chorus_detection.core.model import load_CRNN_model, make_predictions, MODEL_PATH
from chorus_detection.core.utils import get_valid_file_path


def main(input_source: str = None, model_path: str = MODEL_PATH, verbose: bool = True):
    """Process audio and predict chorus locations."""
    try:
        if verbose:
            print("Processing input...")
            
        if not os.path.exists(input_source):
            print(f"Error: File not found at {input_source}")
            return
        audio_path = input_source

        # Process audio
        if verbose:
            print("Processing audio...")
        processed_audio, audio_features = process_audio(audio_path)
        if processed_audio is None:
            print("Failed to process audio. Please try a different file.")
            return

        # Load model
        if verbose:
            print("Loading model...")
        model = load_CRNN_model(model_path=model_path)
        if model is None:
            print("Failed to load model.")
            return

        # Generate predictions
        if verbose:
            print("Making predictions...")
        smoothed_predictions, chorus_start_times, chorus_end_times = make_predictions(
            model, processed_audio, audio_features)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


def run_cli():
    """Entry point for command-line interface."""
    parser = argparse.ArgumentParser(description="Chorus Detection - Identify chorus sections in songs from local audio files or YouTube URLs")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", type=str, help="Path to a local audio file")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help=f"Path to the pretrained model (default: {MODEL_PATH})")
    parser.add_argument("--verbose", action="store_true", help="Verbose output", default=True)
    parser.add_argument("--plot", action="store_true", help="Display plot of the audio waveform", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot display (useful for headless environments)")
    args = parser.parse_args()

    try:
        input_source = args.file
        if not input_source:
            print(parser.format_help())
        else:
            main(input_source, args.model_path, args.verbose)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_cli() 