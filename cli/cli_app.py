#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chorus Detection CLI Tool

This module provides a command-line interface for detecting choruses in audio files,
either from local audio files or YouTube URLs.
"""

import os
# Configure TensorFlow logging before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import sys
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from core.audio_processor import process_audio
from core.model import load_CRNN_model, make_predictions, MODEL_PATH
from core.visualization import plot_predictions, plot_chorus_segments
from core.utils import extract_audio, get_valid_file_path, cleanup_temp_files


def main(input_source: str = None, model_path: str = MODEL_PATH, verbose: bool = True, plot: bool = True):
    """Process audio and predict chorus locations."""
    try:
        if verbose:
            print("Processing input...")
            
        # Determine input type and get audio path
        is_youtube = input_source and input_source.startswith(('http://', 'https://'))
        
        if is_youtube:
            print("Note: YouTube download functionality may be temporarily unavailable due to YouTube's restrictions.")
            print("If download fails, please try using a local audio file instead.")
            print("\nAttempting to extract audio from YouTube...")
            audio_path, video_name = extract_audio(input_source)
            if not audio_path:
                print("Failed to extract audio from the provided URL.")
                return
        else:
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

        # Display visualization
        if plot:
            try:
                print("Displaying plots...")
                plot_predictions(audio_features, smoothed_predictions)
                if chorus_start_times:
                    plot_chorus_segments(audio_features, chorus_start_times, chorus_end_times)
            except Exception as e:
                print(f"Could not display plot: {e}")
                print("Continuing without visualization...")

        # Clean up temporary files
        if is_youtube:
            cleanup_temp_files()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_cli():
    """Entry point for command-line interface."""
    parser = argparse.ArgumentParser(description="Chorus Detection - Identify chorus sections in songs from local audio files or YouTube URLs")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", type=str, help="Path to a local audio file")
    input_group.add_argument("--url", type=str, help="YouTube URL of a song")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help=f"Path to the pretrained model (default: {MODEL_PATH})")
    parser.add_argument("--verbose", action="store_true", help="Verbose output", default=True)
    parser.add_argument("--plot", action="store_true", help="Display plot of the audio waveform", default=True)
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot display (useful for headless environments)")
    args = parser.parse_args()

    try:
        input_source = args.file or args.url
        if not input_source:
            print("\nChorus Detection Tool")
            print("====================")
            
            while True:
                print("\nChoose input method:")
                print("1. Local audio file (from the 'input' folder)")
                print("2. YouTube URL")
                print("3. Exit")
                choice = input("\nEnter choice (1, 2, or 3): ")
                
                if choice == "1":
                    input_source = get_valid_file_path()
                    if input_source:
                        main(input_source, args.model_path, args.verbose, args.plot)
                elif choice == "2":
                    print("\nNote: YouTube download functionality may be temporarily unavailable")
                    print("due to YouTube's restrictions. If download fails, please use a local audio file.")
                    input_source = input("\nPlease enter the YouTube URL of the song: ")
                    main(input_source, args.model_path, args.verbose, args.plot)
                elif choice == "3":
                    print("Exiting program.")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                
                if choice in ["1", "2"]:
                    choice = input("\nDo you want to analyze another file? (y/n): ").lower()
                    if choice != 'y':
                        print("Exiting program.")
                        break
        else:
            main(input_source, args.model_path, args.verbose, args.plot)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_cli() 