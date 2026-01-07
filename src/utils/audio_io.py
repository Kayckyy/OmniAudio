import soundfile as sf
import numpy as np
from pydub import AudioSegment
import os
def load_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.wav', '.flac']:
        data, sr = sf.read(path)
        return data, sr
    else:
        audio = AudioSegment.from_file(path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples /= float(1 << (8 * audio.sample_width - 1))
        if audio.channels == 2: samples = samples.reshape((-1, 2))
        return samples, audio.frame_rate