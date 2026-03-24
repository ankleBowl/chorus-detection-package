from .model import load_CRNN_model, make_predictions, MODEL_PATH
from .audio_processor import process_audio_array, process_audio
import numpy as np
import librosa

class ChorusDetectionModel:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = load_CRNN_model(model_path)

    def predict_file(self, audio_path: str):
        audio, sr = librosa.load(audio_path)
        return self.predict(audio, sr)
        
    def predict(self, audio: np.ndarray, sr: int):
        processed_audio, audio_features = process_audio_array(audio, sr)
        _, chorus_start_times, chorus_end_times = make_predictions(self.model, processed_audio, audio_features)

        out = []
        for i in range(len(chorus_start_times)):
            out.append({
                "start_time": chorus_start_times[i],
                "end_time": chorus_end_times[i],
            })
        return out