AUDIO_PATH = "/Users/ryanpineda/Music/LyePlayer/import/Heather Sommer/On Demand - Single/01 On Demand.m4a"

from chorus_detection import ChorusDetectionModel
import librosa

model = ChorusDetectionModel()

audio, sr = librosa.load(AUDIO_PATH)
out = model.predict(audio, sr)

print(out)