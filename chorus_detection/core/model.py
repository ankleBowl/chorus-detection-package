"""
Model functionality for chorus detection.
"""

import os
import numpy as np
import tensorflow as tf
import librosa

# Set default model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "CRNN", "best_model_V3.h5")


def load_CRNN_model(model_path: str = MODEL_PATH) -> tf.keras.Model:
    """Load a pre-trained CRNN model from the specified path."""
    try:
        # Define custom loss and metrics
        def custom_binary_crossentropy(y_true, y_pred):
            mask = tf.cast(tf.math.not_equal(y_true, -1), tf.float32)
            y_true = tf.maximum(y_true, 0)  # Convert -1 to 0 for BCE calculation
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            masked_bce = bce * mask
            return tf.reduce_sum(masked_bce) / tf.reduce_sum(mask)

        def custom_accuracy(y_true, y_pred):
            mask = tf.cast(tf.math.not_equal(y_true, -1), tf.float32)
            y_true = tf.maximum(y_true, 0)  # Convert -1 to 0 for accuracy calculation
            y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
            correct_predictions = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32) * mask
            return tf.reduce_sum(correct_predictions) / tf.reduce_sum(mask)

        custom_objects = {
            'custom_binary_crossentropy': custom_binary_crossentropy,
            'custom_accuracy': custom_accuracy
        }
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def smooth_predictions(data: np.ndarray) -> np.ndarray:
    """Apply smoothing to model predictions to reduce jitter."""
    # First pass: Moving average
    window_size = 3
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(data), i + window_size // 2 + 1)
        smoothed[i] = np.mean(data[window_start:window_end])

    # Second pass: Eliminate short segments
    min_segment_length = 2
    current_segment_length = 1
    current_value = smoothed[0] > 0.5
    binary_smoothed = np.zeros_like(smoothed, dtype=int)
    binary_smoothed[0] = int(current_value)

    for i in range(1, len(smoothed)):
        new_value = smoothed[i] > 0.5
        if new_value == current_value:
            current_segment_length += 1
        else:
            # If segment is too short, revert to previous value
            if current_segment_length < min_segment_length:
                for j in range(i - current_segment_length, i):
                    binary_smoothed[j] = int(new_value)
            current_value = new_value
            current_segment_length = 1
        binary_smoothed[i] = int(current_value)

    # Third pass: Fix final segment if too short
    if current_segment_length < min_segment_length:
        for j in range(len(smoothed) - current_segment_length, len(smoothed)):
            binary_smoothed[j] = int(not current_value)

    return binary_smoothed


def make_predictions(model, processed_audio, audio_features, torch = True, verbose: bool = False):
    """Make chorus predictions using the loaded model."""
    # Generate predictions
    raw_predictions = model.predict(processed_audio).squeeze()
    
    # Limit predictions to actual meters
    n_meters = min(len(audio_features.meter_grid) - 1, len(raw_predictions))
    predictions = raw_predictions[:n_meters]
    
    # Apply smoothing
    smoothed_predictions = smooth_predictions(predictions)
    
    # Calculate time values for display
    meter_grid_times = librosa.frames_to_time(
        audio_features.meter_grid, sr=audio_features.sr, hop_length=audio_features.hop_length)
    
    # Find chorus segments
    chorus_indices = np.where(smoothed_predictions == 1)[0]
    chorus_start_times = []
    chorus_end_times = []
    
    if len(chorus_indices) > 0:
        # Group consecutive indices
        groups = []
        current_group = [chorus_indices[0]]
        
        for i in range(1, len(chorus_indices)):
            if chorus_indices[i] == chorus_indices[i-1] + 1:
                current_group.append(chorus_indices[i])
            else:
                groups.append(current_group)
                current_group = [chorus_indices[i]]
        groups.append(current_group)
        
        # Display chorus segments
        if verbose:
            print("\nDetected chorus sections:")

        for i, group in enumerate(groups):
            start_time = meter_grid_times[group[0]]
            end_time = meter_grid_times[group[-1] + 1]
            chorus_start_times.append(start_time)
            chorus_end_times.append(end_time)
            
            if verbose:
                start_min, start_sec = divmod(start_time, 60)
                end_min, end_sec = divmod(end_time, 60)

                print(f"Chorus {i+1}: {int(start_min)}:{start_sec:05.2f} - {int(end_min)}:{end_sec:05.2f}")
    else:
        if verbose:
            print("No choruses detected in this audio file.")

    return smoothed_predictions, chorus_start_times, chorus_end_times 


        