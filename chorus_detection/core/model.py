import os
import numpy as np
import librosa

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

# Set default model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "CRNN", "best_model_V3.pth")

class TorchChorusDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1d_0 = nn.Conv1d(in_channels=15, out_channels=128, kernel_size=3, padding='same')
        self.max_pooling1d_0 = nn.MaxPool1d(kernel_size=2)
        self.conv1d_1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.max_pooling1d_1 = nn.MaxPool1d(kernel_size=2)
        self.conv1d_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.max_pooling1d_2 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)

        self.lstm = nn.LSTM(input_size=9728, hidden_size=256, batch_first=True, bidirectional=True)
        self.output = nn.Linear(in_features=512, out_features=1)

    def forward(self, input):
        input = input.permute(0, 1, 3, 2)
        # np.transpose(input_keras, (0, 1, 3, 2))

        batch_count, time, channels, features_x = input.shape
        assert time == 201
        assert channels == 15
        assert features_x == 300

        ic = input.view(batch_count * 201, channels, features_x) 
        ic = self.conv1d_0(ic)
        ic = F.relu(ic)
        ic = self.max_pooling1d_0(ic)
        ic = self.conv1d_1(ic)
        ic = F.relu(ic)
        ic = self.max_pooling1d_1(ic)
        ic = self.conv1d_2(ic)
        ic = F.relu(ic)
        ic = self.max_pooling1d_2(ic)

        ic = ic.view(batch_count, 201, 256, 38)
        ic = ic.permute(0, 1, 3, 2).contiguous()
        ic = ic.view(batch_count, 201, -1)

        # return ic

        mask = (ic.abs().sum(dim=2) != 0)
        lengths = mask.sum(dim=1).int()
        packed_input = pack_padded_sequence(ic, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.output(output)
        output = torch.sigmoid(output)
        return output

def load_CRNN_model(model_path: str = MODEL_PATH):
    model = TorchChorusDetectionModel()
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

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


def make_predictions(model, processed_audio, audio_features, verbose: bool = True):
    """Make chorus predictions using the loaded model."""
    # Generate predictions
    raw_predictions = model(torch.tensor(processed_audio.astype(np.float32))).detach().cpu().numpy().squeeze()
    
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


# from model import load_CRNN_model


# def convert_weights_from_tf():
#     model = load_CRNN_model()
#     dummy_data = np.random.random((1, 201, 300, 15))
#     _ = model.predict(dummy_data)

#     layer_data = {}

#     def recurse(layer, indent=0):
#         # print((" " * (indent * 4)) + layer.name + " - " + str(type(layer)))
#         try:
#             layer_data[layer.name] = layer.get_weights()
#         except:
#             pass

#         try:
#             print((" " * (indent * 4)) + layer.name + " - " + str(layer.output_shape) + " - " + layer.__class__.__name__)
#         except:
#             print((" " * (indent * 4)) + layer.name + " - " + layer.__class__.__name__)
#         if hasattr(layer, "layer"):
#             recurse(layer.layer, indent + 1)
#         if hasattr(layer, "layers"):
#             for nLayer in layer.layers:
#                 recurse(nLayer, indent + 1)

#     for layer in model.layers:
#         recurse(layer)
        
#     pytorch_model = TorchChorusDetectionModel()
#     state_dict = pytorch_model.state_dict()

#     # Start by moving CNN layer weights/biases over
#     state_dict["conv1d_0.weight"] = torch.tensor(layer_data["conv1d"][0]).permute(2, 1, 0)
#     state_dict["conv1d_0.bias"] = torch.tensor(layer_data["conv1d"][1])

#     state_dict["conv1d_1.weight"] = torch.tensor(layer_data["conv1d_1"][0]).permute(2, 1, 0)
#     state_dict["conv1d_1.bias"] = torch.tensor(layer_data["conv1d_1"][1])

#     state_dict["conv1d_2.weight"] = torch.tensor(layer_data["conv1d_2"][0]).permute(2, 1, 0) 
#     state_dict["conv1d_2.bias"] = torch.tensor(layer_data["conv1d_2"][1]) 

#     # Zero one of the pytorch biases for the LSTM
#     state_dict["lstm.bias_ih_l0"] = torch.zeros_like(state_dict["lstm.bias_ih_l0"])
#     state_dict["lstm.bias_ih_l0_reverse"] = torch.zeros_like(state_dict["lstm.bias_ih_l0_reverse"])

#     # Copy the other biases from tensorflow to pytorch
#     state_dict["lstm.bias_hh_l0"] = torch.tensor(layer_data["bidirectional"][2])
#     state_dict["lstm.bias_hh_l0_reverse"] = torch.tensor(layer_data["bidirectional"][5])

#     # Copy LSTM weights
#     state_dict["lstm.weight_ih_l0"] = torch.tensor(layer_data["bidirectional"][0]).permute(1, 0)
#     state_dict["lstm.weight_ih_l0_reverse"] = torch.tensor(layer_data["bidirectional"][3]).permute(1, 0)

#     state_dict["lstm.weight_hh_l0"] = torch.tensor(layer_data["bidirectional"][1]).permute(1, 0)
#     state_dict["lstm.weight_hh_l0_reverse"] = torch.tensor(layer_data["bidirectional"][4]).permute(1, 0)

#     state_dict["output.weight"] = torch.tensor(layer_data["dense"][0]).permute(1, 0)
#     state_dict["output.bias"] = torch.tensor(layer_data["dense"][1])

#     # load back
#     pytorch_model.load_state_dict(state_dict)

#     # save
#     torch.save(pytorch_model.state_dict(), "best_model_V3.pth")

#     # rng = np.random.default_rng()
#     # input_keras = rng.random((1, 201, 300, 15)).astype(np.float32)
#     # input_torch = np.transpose(input_keras, (0, 1, 3, 2))
#     # output = pytorch_model(torch.tensor(input_torch))
#     # tf_output = model.predict(input_keras, verbose=0).squeeze()

#     # print(output.mean())
#     # print(tf_output.mean())

# if __name__ == "__main__":
#     convert_weights_from_tf()