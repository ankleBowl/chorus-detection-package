"""
Utility functions for chorus detection.
"""

import os
import shutil
from functools import reduce
from typing import Optional, Tuple

# Constants
AUDIO_TEMP_PATH = "output/temp"

def get_valid_file_path() -> Optional[str]:
    """List audio files in the data/processed folder and let the user select one."""
    input_folder = "data/processed"
    valid_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
    
    # Find audio files
    audio_files = []
    if os.path.exists(input_folder) and os.path.isdir(input_folder):
        for file in os.listdir(input_folder):
            _, ext = os.path.splitext(file)
            if ext.lower() in valid_extensions:
                audio_files.append(file)
    
    if not audio_files:
        print("\nNo audio files found in the data/processed folder.")
        print("Please copy your audio files to the 'data/processed' folder.")
        print("Supported formats: .mp3, .wav, .ogg, .flac, .m4a")
        return None
    
    # Display files and get selection
    total_files = len(audio_files)
    max_display = 20  # Maximum number of files to display at once
    
    print(f"\nFound {total_files} audio files in the data/processed folder.")
    print("Choose a file by:")
    print("- Entering a file number from the list below")
    print("- Typing 'name:' followed by the filename (e.g., 'name:song.mp3')")
    print("- Typing 'list' to show more files")
    print("- Typing 'back' to return to the main menu")
    
    start_idx = 0
    while True:
        # Display a subset of files
        end_idx = min(start_idx + max_display, total_files)
        for i in range(start_idx, end_idx):
            print(f"{i+1}. {audio_files[i]}")
        
        if total_files > max_display:
            remaining = total_files - end_idx
            if remaining > 0:
                print(f"... and {remaining} more files. Type 'list' to see more.")
        
        selection = input("\nEnter your selection:\n> ")
        
        if selection.lower() == 'back':
            return None
        elif selection.lower() == 'list':
            start_idx = (start_idx + max_display) % total_files
            continue
        elif selection.lower().startswith('name:'):
            # Direct filename input
            filename = selection[5:].strip()
            if filename in audio_files:
                return os.path.join(input_folder, filename)
            else:
                # Check if it's a partial match
                matches = [file for file in audio_files if filename.lower() in file.lower()]
                if len(matches) == 1:
                    return os.path.join(input_folder, matches[0])
                elif len(matches) > 1:
                    print("\nMultiple matches found:")
                    for i, match in enumerate(matches, 1):
                        print(f"{i}. {match}")
                    sub_selection = input("\nEnter the number of the file you want to analyze:\n> ")
                    try:
                        index = int(sub_selection) - 1
                        if 0 <= index < len(matches):
                            return os.path.join(input_folder, matches[index])
                        else:
                            print(f"Error: Please enter a number between 1 and {len(matches)}.")
                    except ValueError:
                        print("Error: Please enter a valid number.")
                else:
                    print(f"Error: No file named '{filename}' found.")
        else:
            # Number selection
            try:
                index = int(selection) - 1
                if 0 <= index < total_files:
                    return os.path.join(input_folder, audio_files[index])
                else:
                    print(f"Error: Please enter a number between 1 and {total_files}.")
            except ValueError:
                print("Error: Please enter a valid selection.")


def cleanup_temp_files():
    """Clean up temporary files after processing."""
    if os.path.exists(AUDIO_TEMP_PATH):
        try:
            shutil.rmtree(AUDIO_TEMP_PATH)
        except Exception as e:
            print(f"Warning: Could not clear temporary files: {e}") 