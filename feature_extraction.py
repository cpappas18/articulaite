import warnings
import librosa
import surfboard
from surfboard.sound import Waveform
import Signal_Analysis.features.signal
import sys


warnings.filterwarnings('ignore')


def extract_features(sound_wav_file):
    """
    Extracts audio features from a .wav file

    Args:
        sound_wav_file: audio file to extract features from

    Returns: list of extracted audio features in the format [shimmers, jitters, HNR, DFA, ]

    """
    # converting audio file into a vector and getting the sample rate
    audio, sr = librosa.load(sound_wav_file, sr=None)

    # converting our audio vector into a waveform object
    sound = Waveform(signal=audio, sample_rate=sr)

    # redirecting irrelevant output from Signal_Analysis methods below
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')

    # extracting features
    features = {}
    shimmers = sound.shimmers()
    jitters = Signal_Analysis.features.signal.get_Jitter(audio, sr)
    HNR_value = Signal_Analysis.features.signal.get_HNR(audio, sr, time_step=0, min_pitch=75, silence_threshold=0.1, periods_per_window=4.5)
    dfa = surfboard.dfa.get_dfa(audio, [1024]) # 1024 is the default window_length
    F_0 = Signal_Analysis.features.signal.get_F_0(audio, sr, time_step=0, min_pitch=75) # Fundamental Mean Frequency

    sys.stdout = save_stdout

    features.update(shimmers)
    jitters["localJitter"] = jitters.pop("local")
    jitters["localAbsJitter"] = jitters.pop("local, absolute")
    features.update(jitters)
    features["HNR"] = HNR_value
    features["DFA"] = dfa
    features["Fundamental Mean Frequency"] = F_0[0]

    return features