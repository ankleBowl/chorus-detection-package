from model import load_CRNN_model

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

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

def convert_weights_from_tf():
    model = load_CRNN_model()
    dummy_data = np.random.random((1, 201, 300, 15))
    _ = model.predict(dummy_data)

    layer_data = {}

    def recurse(layer, indent=0):
        # print((" " * (indent * 4)) + layer.name + " - " + str(type(layer)))
        try:
            layer_data[layer.name] = layer.get_weights()
        except:
            pass

        try:
            print((" " * (indent * 4)) + layer.name + " - " + str(layer.output_shape) + " - " + layer.__class__.__name__)
        except:
            print((" " * (indent * 4)) + layer.name + " - " + layer.__class__.__name__)
        if hasattr(layer, "layer"):
            recurse(layer.layer, indent + 1)
        if hasattr(layer, "layers"):
            for nLayer in layer.layers:
                recurse(nLayer, indent + 1)

    for layer in model.layers:
        recurse(layer)
        
    pytorch_model = TorchChorusDetectionModel()
    state_dict = pytorch_model.state_dict()

    # Start by moving CNN layer weights/biases over
    state_dict["conv1d_0.weight"] = torch.tensor(layer_data["conv1d"][0]).permute(2, 1, 0)
    state_dict["conv1d_0.bias"] = torch.tensor(layer_data["conv1d"][1])

    state_dict["conv1d_1.weight"] = torch.tensor(layer_data["conv1d_1"][0]).permute(2, 1, 0)
    state_dict["conv1d_1.bias"] = torch.tensor(layer_data["conv1d_1"][1])

    state_dict["conv1d_2.weight"] = torch.tensor(layer_data["conv1d_2"][0]).permute(2, 1, 0) 
    state_dict["conv1d_2.bias"] = torch.tensor(layer_data["conv1d_2"][1]) 

    # Zero one of the pytorch biases for the LSTM
    state_dict["lstm.bias_ih_l0"] = torch.zeros_like(state_dict["lstm.bias_ih_l0"])
    state_dict["lstm.bias_ih_l0_reverse"] = torch.zeros_like(state_dict["lstm.bias_ih_l0_reverse"])

    # Copy the other biases from tensorflow to pytorch
    state_dict["lstm.bias_hh_l0"] = torch.tensor(layer_data["bidirectional"][2])
    state_dict["lstm.bias_hh_l0_reverse"] = torch.tensor(layer_data["bidirectional"][5])

    # Copy LSTM weights
    state_dict["lstm.weight_ih_l0"] = torch.tensor(layer_data["bidirectional"][0]).permute(1, 0)
    state_dict["lstm.weight_ih_l0_reverse"] = torch.tensor(layer_data["bidirectional"][3]).permute(1, 0)

    state_dict["lstm.weight_hh_l0"] = torch.tensor(layer_data["bidirectional"][1]).permute(1, 0)
    state_dict["lstm.weight_hh_l0_reverse"] = torch.tensor(layer_data["bidirectional"][4]).permute(1, 0)

    state_dict["output.weight"] = torch.tensor(layer_data["dense"][0]).permute(1, 0)
    state_dict["output.bias"] = torch.tensor(layer_data["dense"][1])

    # load back
    pytorch_model.load_state_dict(state_dict)

    # save
    torch.save(pytorch_model.state_dict(), "best_model_V3.pth")

    # rng = np.random.default_rng()
    # input_keras = rng.random((1, 201, 300, 15)).astype(np.float32)
    # input_torch = np.transpose(input_keras, (0, 1, 3, 2))
    # output = pytorch_model(torch.tensor(input_torch))
    # tf_output = model.predict(input_keras, verbose=0).squeeze()

    # print(output.mean())
    # print(tf_output.mean())

if __name__ == "__main__":
    convert_weights_from_tf()